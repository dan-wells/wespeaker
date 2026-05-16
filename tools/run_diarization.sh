#!/bin/bash
# Copyright (c) 2022-2023 Xu Xiang
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

. ./path.sh || exit 1

set -euo pipefail

stage=-1
stop_stage=-1
sad_type="oracle"       # oracle/system
assign_type="cluster"   # cluster/identify/beamform
cluster_type="spectral" # spectral/umap
beamform_mode="supervised"  # supervised/unsupervised
beamform_n_workers=4

emb_window=1.5
emb_stride=0.75
subseg_cmn=true  # do cmn on the sub-segment (causal) or on the vad segment (non-causal)
utt2num_spks=""  # oracle number of speakers per file for spectral clustering
get_each_file_res=1

pretrained_model=pretrained_models/voxceleb_resnet34_LM.onnx
enrol_scp=""
data_dir=""
exp_label=""
assign_threshold=""  # min cosine similarity for speaker assignment; omit low-confidence subsegments
score_overlap=true  # if false, strip overlap regions from ref before scoring (hyp speech in overlap = false alarm)
map_spk_ids=false   # map cluster labels to speaker IDs after RTTM writing

. tools/parse_options.sh

if [ -z "$data_dir" ]; then
    echo "Error: --data_dir is required"
    exit 1
fi

exp="exp"

# Build exp_label from parameters if not provided
if [ -z "$exp_label" ]; then
    if [ "$assign_type" == "cluster" ]; then
        assign_tag="${cluster_type}_cluster"
    elif [ "$assign_type" == "identify" ]; then
        assign_tag="identify"
    elif [ "$assign_type" == "beamform" ]; then
        assign_tag="beamform_${beamform_mode}"
    else
        echo "Error: unknown assign_type '${assign_type}'"
        exit 1
    fi
    exp_label="${assign_tag}_win_${emb_window}_stride_${emb_stride}_${sad_type}_sad"
fi
exp="${exp}/${exp_label}"

mkdir -p ${data_dir}/${exp}

# Prerequisite
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # [1] Download evaluation toolkit
    mkdir -p external_tools
    wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
    unzip -o external_tools/SCTK-v2.4.12.zip -d external_tools

    # [2] Download ResNet34 speaker model pretrained by WeSpeaker Team
    mkdir -p pretrained_models
    wget -c https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx -O pretrained_models/voxceleb_resnet34_LM.onnx
fi

# Prepare meeting_sim data
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # Prepare wav.scp
    find ${data_dir}/ -name "mixed.wav" | sort -u | while read -r path; do
        utt=$(basename $(dirname ${path}))
        echo "${utt} ${path}"
    done > ${data_dir}/wav.scp

    # Prepare multichannel wav.scp (for beamform mode)
    find ${data_dir}/ -name "multichannel.wav" | sort -u | while read -r path; do
        utt=$(basename $(dirname ${path}))
        echo "${utt} ${path}"
    done > ${data_dir}/wav_multichannel.scp

    # Prepare RTTM for oracle SAD and scoring
    mkdir -p ${data_dir}/${exp}/ref_rttm
    while read -r utt wav_path; do
        ln -sf ${data_dir}/meetings/${utt}/meeting.rttm ${data_dir}/${exp}/ref_rttm/${utt}.rttm
    done < ${data_dir}/wav.scp
fi

# Voice activity detection
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # Set VAD min duration
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM
        while read -r utt wav_path; do
            python3 wespeaker/diar/make_oracle_sad.py \
                    --rttm ${data_dir}/${exp}/ref_rttm/${utt}.rttm \
                    --min-duration $min_duration
        done < ${data_dir}/wav.scp > ${data_dir}/${exp}/oracle_sad
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
       # System SAD: applying 'silero' VAD
       python3 wespeaker/diar/make_system_sad.py \
               --scp ${data_dir}/wav.scp \
               --min-duration $min_duration > ${data_dir}/${exp}/system_sad
    fi
fi


# Extract fbank features (skip for beamform mode)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ] && [ "$assign_type" != "beamform" ]; then

    [ -d "${data_dir}/${exp}/${sad_type}_sad_fbank" ] && rm -r ${data_dir}/${exp}/${sad_type}_sad_fbank

    echo "================================================================================"
    echo "Make Fbank features and store it under ${data_dir}/${exp}/${sad_type}_sad_fbank"
    echo "================================================================================"
    mkdir -p ${data_dir}/${exp}/${sad_type}_sad_fbank
    python3 wespeaker/diar/make_fbank.py \
        --scp ${data_dir}/wav.scp \
        --segments ${data_dir}/${exp}/${sad_type}_sad \
        --ark-path ${data_dir}/${exp}/${sad_type}_sad_fbank/fbank.ark \
        --subseg-cmn ${subseg_cmn}
fi

# Extract embeddings (skip for beamform mode)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ] && [ "$assign_type" != "beamform" ]; then

    [ -d "${data_dir}/${exp}/${sad_type}_sad_embedding" ] && rm -r ${data_dir}/${exp}/${sad_type}_sad_embedding

    echo "================================================================================"
    echo "Extract embeddings and store it under ${exp}/${sad_type}_sad_embedding"
    echo "================================================================================"
    mkdir -p ${data_dir}/${exp}/${sad_type}_sad_embedding
    python3 wespeaker/diar/extract_emb.py \
        --scp ${data_dir}/${exp}/${sad_type}_sad_fbank/fbank.scp \
        --ark-path ${data_dir}/${exp}/${sad_type}_sad_embedding/emb.ark \
        --source ${pretrained_model} \
        --device cuda \
        --batch-size 96 \
        --frame-shift 10 \
        --window-secs ${emb_window} \
        --period-secs ${emb_stride} \
        --subseg-cmn ${subseg_cmn}
fi


# Speaker assignment (cluster / identify / beamform)
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    labels_file="${data_dir}/${exp}/${sad_type}_sad_labels"
    [ -f "${labels_file}" ] && rm "${labels_file}"

    echo "================================================================================"
    echo "Assigning speakers (${assign_type}) -> ${labels_file}"
    echo "================================================================================"

    if [ "$assign_type" == "cluster" ]; then
        python3 wespeaker/diar/${cluster_type}_clusterer.py \
                --scp ${data_dir}/${exp}/${sad_type}_sad_embedding/emb.scp \
                --output ${labels_file} \
                ${utt2num_spks:+--utt2num_spks ${utt2num_spks}} \
                ${assign_threshold:+--threshold ${assign_threshold}}

    elif [ "$assign_type" == "identify" ]; then
        if [ -z "$enrol_scp" ]; then
            echo "Error: --enrol_scp is required for assign_type=identify"
            exit 1
        fi
        python3 wespeaker/diar/identify.py \
                --scp ${data_dir}/${exp}/${sad_type}_sad_embedding/emb.scp \
                --enrol-scp ${enrol_scp} \
                --metadata-dir ${data_dir}/meetings \
                --output ${labels_file} \
                ${assign_threshold:+--threshold ${assign_threshold}}

    elif [ "$assign_type" == "beamform" ]; then
        beamform_args="--wav-scp ${data_dir}/wav_multichannel.scp \
                --segments ${data_dir}/${exp}/${sad_type}_sad \
                --output ${labels_file} \
                --mode ${beamform_mode} \
                --window-secs ${emb_window} \
                --period-secs ${emb_stride} \
                --n-workers ${beamform_n_workers}"
        if [ "$beamform_mode" == "supervised" ]; then
            beamform_args="${beamform_args} --metadata-dir ${data_dir}/meetings"
        fi
        python3 wespeaker/diar/beamform_diar.py ${beamform_args}
    fi
fi


# Convert labels to RTTMs
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    python3 wespeaker/diar/make_rttm.py \
            --labels ${data_dir}/${exp}/${sad_type}_sad_labels \
            --channel 1 > ${data_dir}/${exp}/${sad_type}_sad_rttm

    if [ "${map_spk_ids}" == "true" ]; then
        hyp_rttm="${data_dir}/${exp}/${sad_type}_sad_rttm"
        mapped_rttm="${data_dir}/${exp}/${sad_type}_sad_rttm_mapped"
        if [ -n "${enrol_scp}" ]; then
            python3 wespeaker/diar/map_speakers.py \
                --hyp-rttm "${hyp_rttm}" \
                --output "${mapped_rttm}" \
                --enrol-scp "${enrol_scp}" \
                --emb-scp "${data_dir}/${exp}/${sad_type}_sad_embedding/emb.scp"
        else
            python3 wespeaker/diar/map_speakers.py \
                --hyp-rttm "${hyp_rttm}" \
                --output "${mapped_rttm}" \
                --ref-rttm <(cat "${data_dir}/${exp}/ref_rttm/"*.rttm)
        fi
    fi
fi


# Evaluate the result
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    ref_dir=${data_dir}/${exp}/ref_rttm
    echo "================================================================================"
    echo "Compute DER results"
    echo "================================================================================"
    perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
         -c 0.25 \
         -r <(cat ${ref_dir}/*.rttm) \
         -s ${data_dir}/${exp}/${sad_type}_sad_rttm 2>&1 | tee ${data_dir}/${exp}/${sad_type}_sad_res

    if [ ${get_each_file_res} -eq 1 ];then
        single_file_res_dir=${data_dir}/${exp}/${sad_type}_single_file_res
        mkdir -p $single_file_res_dir
        echo "Compute per-file DER results, stored under ${single_file_res_dir}"

        awk '{print $2}' ${data_dir}/${exp}/${sad_type}_sad_rttm | sort -u  | while read file_name; do
            perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
                 -c 0.25 \
                 -r <(cat ${ref_dir}/${file_name}.rttm) \
                 -s <(grep "${file_name}" ${data_dir}/${exp}/${sad_type}_sad_rttm) > ${single_file_res_dir}/${file_name}_res
        done
        echo "Done!"
    fi

    if [ "${score_overlap}" == "false" ]; then
        stripped_ref_dir=${data_dir}/${exp}/ref_rttm_no_overlap
        mkdir -p ${stripped_ref_dir}

        echo "================================================================================"
        echo "Stripping overlap from reference RTTMs -> ${stripped_ref_dir}"
        echo "================================================================================"
        for rttm_path in ${ref_dir}/*.rttm; do
            file_name=$(basename ${rttm_path} .rttm)
            python3 wespeaker/diar/strip_overlap.py \
                --rttm ${rttm_path} \
                --output ${stripped_ref_dir}/${file_name}.rttm
        done

        echo "================================================================================"
        echo "Compute DER results (overlap regions treated as silence in reference)"
        echo "================================================================================"
        perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
             -c 0.25 \
             -r <(cat ${stripped_ref_dir}/*.rttm) \
             -s ${data_dir}/${exp}/${sad_type}_sad_rttm 2>&1 \
             | tee ${data_dir}/${exp}/${sad_type}_sad_res_no_overlap

        if [ ${get_each_file_res} -eq 1 ]; then
            single_file_res_no_overlap_dir=${data_dir}/${exp}/${sad_type}_single_file_res_no_overlap
            mkdir -p ${single_file_res_no_overlap_dir}
            echo "Compute per-file DER results (no overlap), stored under ${single_file_res_no_overlap_dir}"

            awk '{print $2}' ${data_dir}/${exp}/${sad_type}_sad_rttm | sort -u | while read file_name; do
                perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
                     -c 0.25 \
                     -r <(cat ${stripped_ref_dir}/${file_name}.rttm) \
                     -s <(grep "${file_name}" ${data_dir}/${exp}/${sad_type}_sad_rttm) \
                     > ${single_file_res_no_overlap_dir}/${file_name}_res
            done
            echo "Done!"
        fi
    fi
fi
