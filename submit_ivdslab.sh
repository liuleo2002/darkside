#!/bin/bash
# Set the file path
build_dir=$(pwd)
sim_name="PMMA"

base_dir="/data/darkside/simulation/B9_simulation/NeutronAnalysis/g4ds_clusters/build/TPC_PDU_N" # path for .fil files
output_base="/data/darkside/simulation/B9_simulation/LeoDCR/final" # path for where output .slc files should be stored
csv_base="/data/darkside/simulation/B9_simulation/NeutronAnalysis/g4ds_clusters/build/passed_sims" # read csv to find good sims to run

Nsub=10

for ((n_loc=0; n_loc < Nsub; n_loc++)); do
    input_dir="${base_dir}/PMMA_${n_loc}/output/fil_log_dep"
    dir="${output_base}/${sim_name}_${n_loc}"
    cd "${output_base}"
    rm -rf "${dir}"
    mkdir -p "${dir}"
    cd "${dir}" 
    mkdir -p slices
    mkdir -p log
    
    get_the_name="${dir##*/}"
    csv_file="${csv_base}/${get_the_name}.csv"
    
    skip_header=true
    files=()
    while IFS=',' read -r names; do
        # If the first line is the header, skip it
        if [ "$skip_header" = true ]; then
            skip_header=false
            continue
        fi
        # Process each row
        result=$(echo "$names" | awk -F, '{print $2}')
        files+=("$result")
    done < "$csv_file"

    cat << EOF > submitter_ivdslab.sub
universe = vanilla
getenv = true
request_memory = 5 GB
request_disk = 10 GB
arguments = \$(ProcId) \$(ClusterId)

executable = job_ivdslab.sh
log = log_root_\$(ClusterId).sh
error = ${dir}/log/output.\$(ProcId)

ShouldTransferFiles = YES
WhenToTransferOutput = ON_EXIT

EOF
    for (( i=0; i<${#files[@]}; i++ )); do
        echo "arguments = $i \$(ClusterId) ${files[$i]}" >> submitter_ivdslab.sub
        echo "queue" >> submitter_ivdslab.sub
    done

    chmod +x submitter_ivdslab.sub

    cat << EOF > job_ivdslab.sh
#!/bin/bash
PROCID=\$1
CLUSID=\$2
file_name=\$3

cd ${build_dir}
echo "The file to be submitted is \${file_name}"

# Run daq_slices_sweep.py to create slice files for each DCR value. Replace python script path with location of your daq_slices_sweep.py
python -u /home/weihengliu/B9_simulations/iv-dslab/exe/daq_slices_sweep.py -i ${input_dir}/\${file_name}.fil -o ${dir}/slices/\${file_name}.slc --dcr 40,500,700

EOF

    chmod +x job_ivdslab.sh
    condor_submit submitter_ivdslab.sub
done
