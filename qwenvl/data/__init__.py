import re

# Define placeholders for dataset paths
data_dict = {
    "0_30_s_academic_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/0_30_s_academic_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "0_30_s_youtube_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/0_30_s_youtube_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "0_30_s_activitynetqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/0_30_s_activitynetqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "0_30_s_nextqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/0_30_s_nextqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "0_30_s_perceptiontest": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/0_30_s_perceptiontest", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "30_60_s_academic_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/30_60_s_academic_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "30_60_s_youtube_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/30_60_s_youtube_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "30_60_s_activitynetqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/30_60_s_activitynetqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "30_60_s_nextqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/30_60_s_nextqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "30_60_s_perceptiontest": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/30_60_s_perceptiontest", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "1_2_m_academic_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/1_2_m_academic_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "1_2_m_youtube_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/1_2_m_youtube_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "1_2_m_activitynetqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/1_2_m_activitynetqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "1_2_m_nextqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/1_2_m_nextqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "2_3_m_academic_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/2_3_m_academic_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "2_3_m_youtube_v0_1": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/2_3_m_youtube_v0_1", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "2_3_m_activitynetqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/2_3_m_activitynetqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
    "2_3_m_nextqa": {"annotation_dir": "/blob/dyb/LLaVA-Video-178K/2_3_m_nextqa", "data_dir": "/blob/dyb/LLaVA-Video-178K"},
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
