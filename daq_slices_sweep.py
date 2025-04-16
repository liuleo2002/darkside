"""
DS-20k G4DS Time Slice Processing with Parameter Sweeping

This script processes time slices produced by the G4DS simulation for the DS-20k detector.
It performs electronics simulation, zero-length encoding (ZLE), and hit finding on photoelectron
(PE) data. The script extends the original daq_slices.py by adding parameter sweeping capabilities,
allowing users to process the same input file with multiple values of parameters like SNR, sampling
rate, or jitter via command-line arguments.

Usage:
    python daq_slices.py -i input_file (.fil format) -o output_base (.slc format) [--snr 10,20,30] [--jitter 5,10,15] ect.

Dependencies:
    - numpy
    - ROOT (with PyROOT)
    - dsutils, dselec, dsio, dsdetector (from iv-dslab repo)
    - pathlib
    - argparse
    - logging
"""
import argparse as ap
import logging
import numpy as np
import sys
from pathlib import Path
from dsutils import nu, set_random_seeds, concat_recarrays
from dselec import *
from dsio import *
from dsdetector import tpc_chmap
import ROOT
from ROOT import TH1F, TFile

logger = logging.getLogger(__name__)

def process_event(i, ev, out):
    """
    Process a single TPC event by simulating electronics, finding ZLEs, and identifying hits
    Same logic as daq_slices.py

    Inputs:
        i (int): Event index
        ev (dict): Input event dictionary of PE data
        out (dict): Output event dictionary to store results

    Returns:
        bool: True if valid hits and ZLEs are found and written to 'out', False otherwise.
    """
    hits, zles = [], []
    summed_wfs = None
    t_starts, gates = find_waveform_gates(ev['pe'])
    out['pe'] = add_noise_pes(ev['pe'], elec_cfg.get('daq.slice')) # add noise to PE data
    add_daq_jitter(out['pe'])
    
    for t_start, gate in zip(t_starts, gates):
        try:
            pes = out['pe'][(out['pe']['time'] >= t_start) & (out['pe']['time'] <= t_start + gate)]
            logger.debug(f"Simulating waveform for t_start={t_start / nu.ms:.3f} ms, gate={gate / nu.us:.3f} us, NPE={len(pes)}")
            wfs = create_veto_waveforms(pes, gate, t_start)
            z = find_zle_intervals(wfs, t_start)
            
            # Add error handling as sometimes slices fail due extreme parameter configuration (eg. very low snr)
            try:
                summed_wfs = sum_zles(z, wfs, summed_wfs, t_start)
            except ValueError as e:
                if "ZLE interval too short" in str(e):
                    logger.warning(f"Event {i}: ZLE interval too short for baseline subtraction. This may happen with low SNR values. Skipping waveform summation.")
                    continue
                else:
                    raise e
                    
            h = find_hits(wfs, z, t_start)
            h['zle_id'] += sum(len(_z) for _z in zles)
            zles.append(z)
            hits.append(h)
            
            # Save waveform data
            if summed_wfs is not None:
                out['top_wf'], out['bot_wf'] = downsample_summed_wfs(summed_wfs)
            
        except Exception as e:
            logger.warning(f"Event {i}: Error processing waveform at t_start={t_start / nu.ms:.3f} ms: {str(e)}")
            continue

    try:
        pes = get_pes_outside_gates(out['pe'], t_starts, gates)
        z, h = find_effective_zles_hits(pes)
        h['zle_id'] += sum(len(_z) for _z in zles)
        zles.append(z)
        hits.append(h)
    except Exception as e:
        logger.warning(f"Event {i}: Error processing PEs outside gates: {str(e)}")

    # Only proceed if we have some valid data
    if len(hits) > 0 and len(zles) > 0:
        out['zles'] = concat_recarrays(zles) # combine zle data
        out['hits'] = concat_recarrays(hits) # combine hit data
        out['zles']['integral'] /= get_zle_gain_veto()['integral_mean']
        for key in ['integral', 'max']:
            out['hits'][key] /= get_hit_gain_veto()[f'{key}_mean']
        out['hits'].sort(order=['zle_id', 'sample'])
        return True
    else:
        logger.warning(f"Event {i}: No valid hits or ZLEs found. This event will be skipped.")
        return False

def update_config_parameter(section, param, value):
    """
    Update a specific configuration parameter in the electronics configuration.

    Inputs:
        section (str): Configuration section from dselec.ini file (e.g., 'daq', 'sipm', 'zle').
        param (str): Parameter name used in dselec.ini file (e.g., 'snr', 'jitter').
        value: Parameter value you want to use.
    """
    if section == 'daq':
        if param == 'snr':
            elec_cfg.set('daq.snr', float(value))
        elif param == 'sampling':
            elec_cfg.set('daq.sampling', float(value) * nu.MHz)
        elif param == 'jitter':
            elec_cfg.set('daq.jitter', float(value) * nu.ns)
        elif param == 'noise_spectrum':
            elec_cfg.set('daq.noise_spectrum', str(value))
    elif section == 'sipm':
        if param == 'dcr':
            elec_cfg.set('sipm.dcr', float(value) * nu.Hz)
        elif param == 'tau':
            elec_cfg.set('sipm.tau', float(value) * nu.ns)
    elif section == 'zle':
        if param in ['pre_threshold', 'post_threshold']:
            elec_cfg.set(f'zle.{param}', float(value))
        elif param in ['pre_trigger', 'post_trigger']:
            elec_cfg.set(f'zle.{param}', int(value))

def process_with_params(input_path, output_base, section, param, values, **kwargs):
    """
    Process the input file with different parameter values, saving results to separate output files.

    Inputs:
        input_path (str): Path to the input file (.fil) files.
        output_base (str): Base name for output files.
        section (str): Configuration section to modify (e.g., 'daq'). Refer to dselec.ini
        param (str): Parameter to sweep (e.g., 'snr'). Refer to dselec.ini
        values (list): List of parameter values to sweep over.

    Outputs:
        - Output files are named as [output_base]_[section]_[param]_[value].slc.
    """
    # Create output directory
    output_dir = Path(output_base).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name
    base_name = Path(output_base).stem
    extension = Path(output_base).suffix
    
    # Process for each parameter value
    for value in values:
        try:
            # Update configuration
            update_config_parameter(section, param, value)
            
            # Create output filename with parameter value
            output_path = output_dir / f"{base_name}_{section}_{param}_{value}{extension}"
            
            # Process file
            logger.info(f"Processing with {section}.{param} = {value}")
            fin = G4DSBinaryReader(input_path, **kwargs)
            fout = SliceWriter(str(output_path), header=fin.header)
            
            valid_events = 0
            for i, ev in enumerate(fin):
                ev['pe'] = ev['veto_pe']
                logger.debug(f'Processing event {i}')
                out = fout.create_empty_event()
                ev['pe']['time'] += elec_cfg.get('daq.offset')
                if process_event(i, ev, out):
                    fout.write(out)
                    valid_events += 1
            
            fin.close()
            fout.close()
            
            if valid_events > 0:
                logger.info(f"Saved output to {output_path} ({valid_events} valid events)")
            else:
                logger.warning(f"No valid events processed for {section}.{param} = {value}")
                # Remove empty output files
                if output_path.exists():
                    output_path.unlink()
                
        except Exception as e:
            logger.error(f"Error processing {section}.{param} = {value}: {str(e)}")
            continue

def parse_args(argv=None):
    """
    Parse command-line arguments

    Inputs:
        argv (list): List of command-line arguments. Defaults to sys.argv[1:].

    Outputs:
        dict: Dictionary of parameters
    """
    if argv is None:
        argv = sys.argv[1:]
    # same as original daq_slices.py    
    parser = ap.ArgumentParser(description='DS-20k G4DS time slice processing with parameter sweeping')
    parser.add_argument('-i', '--input', dest='input_path', required=True, help='Input file name.')
    parser.add_argument('-o', '--output', dest='output_base', required=True, help='Base output file name.')
    parser.add_argument('--start', default=ap.SUPPRESS, type=int, help='Event to start at.')
    parser.add_argument('--stop', default=ap.SUPPRESS, type=int, help='Event to stop at.')
    parser.add_argument('-s', '--seeds', type=int, nargs=2, default=[1234, 1235], help='Numpy and numba random seeds')
    
    # Add parameter sweep arguments based on dselec.ini
    parser.add_argument('--snr', type=str, help='SNR values to sweep (comma-separated)')
    parser.add_argument('--sampling', type=str, help='Sampling rate values in MHz to sweep (comma-separated)')
    parser.add_argument('--jitter', type=str, help='Jitter values in ns to sweep (comma-separated)')
    parser.add_argument('--dcr', type=str, help='Dark count rate values in Hz to sweep (comma-separated)')
    parser.add_argument('--tau', type=str, help='Tau values in ns to sweep (comma-separated)')
    parser.add_argument('--pre-threshold', type=str, help='ZLE pre-threshold values to sweep (comma-separated)')
    parser.add_argument('--post-threshold', type=str, help='ZLE post-threshold values to sweep (comma-separated)')
    parser.add_argument('--pre-trigger', type=str, help='ZLE pre-trigger values to sweep (comma-separated)')
    parser.add_argument('--post-trigger', type=str, help='ZLE post-trigger values to sweep (comma-separated)')
    parser.add_argument('--noise-spectrum', type=str, help='Noise spectrum files to use (comma-separated)')
    
    args = vars(parser.parse_args(argv))
    
    set_random_seeds(*args.pop('seeds'))
    
    return args

def main(**kwargs):
    """
    Main function to handle parameter sweeping and processing of G4DS time slices.

    Notes:
        - If a sweep parameter is specified (e.g., --snr), the script processes the input file for each value.
        - If no sweep parameter is specified, it processes with the default SNR value in dselec.ini.
    """
    input_path = kwargs.pop('input_path')
    output_base = kwargs.pop('output_base')
    
    # Parameter mapping
    param_mapping = {
        'snr': ('daq', 'snr'),
        'sampling': ('daq', 'sampling'),
        'jitter': ('daq', 'jitter'),
        'noise_spectrum': ('daq', 'noise_spectrum'),
        'dcr': ('sipm', 'dcr'),
        'tau': ('sipm', 'tau'),
        'pre_threshold': ('zle', 'pre_threshold'),
        'post_threshold': ('zle', 'post_threshold'),
        'pre_trigger': ('zle', 'pre_trigger'),
        'post_trigger': ('zle', 'post_trigger')
    }
    
    # Find which parameter to sweep
    for param_name, param_value in list(kwargs.items()):
        if param_value is not None and param_name in param_mapping:
            section, param = param_mapping[param_name]
            # Special handling for noise_spectrum as it's a string
            if param == 'noise_spectrum':
                values = [x.strip() for x in param_value.split(',')]
            else:
                values = [float(x.strip()) for x in param_value.split(',')]
            logger.info(f"Sweeping {section}.{param} over values: {values}")
            process_with_params(input_path, output_base, section, param, values)
            return
    
    # If no parameter sweep specified, just process normally
    process_with_params(input_path, output_base, 'daq', 'snr', [elec_cfg.get('daq.snr')])

if __name__ == '__main__':
    from dsutils import setup_logging
    setup_logging(level='debug')
    kwargs = parse_args()
    main(**kwargs) 
