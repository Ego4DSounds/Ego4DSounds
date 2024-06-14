# Ego4DSounds
Ego4DSounds is a subset of Ego4D, an existing large-scale egocentric video dataset. Videos have a high action-audio correspondence, making it a high-quality dataset for action-to-sound generation.

[Explore the dataset](https://ego4dsounds.github.io/)

## Action2Sound

Dataset introduced in _"Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos"_.

Action2Sound is an ambient-aware approach that disentangles the action sound from the ambient sound, allowing successful generation after training with diverse in-the-wild data, as well as controllable conditioning on ambient sound levels.

![action2sound](https://github.com/Ego4DSounds/Ego4DSounds/assets/59634524/40a9d037-9134-4edc-82a5-1d81c6bbb40c)

[Explore the project](https://vision.cs.utexas.edu/projects/action2sound/)

## Contents

This repository contains scripts for processing the Ego4DSounds dataset. It includes functionality for loading video and audio data and extracting clips using metadata.

- `extract_ego4d_clips.py`: Extracts clips from the Ego4D dataset
- `dataset.py`: Defines the Ego4DSounds dataset class for loading and processing video and audio clips
- Metadata files: `train_clips_1.2m.csv`, `test_clips_11k.csv`, `ego4d.json`

Each row in the csv files has the following columns
```
video_uid, video_dur, narration_source, narration_ind, narration_time, clip_start, clip_end, clip_text, tag_verb, tag_noun, positive, clip_file, speech, background_music, traffic_noise, wind_noise
```

## BibTeX
```
@article{chen2024action2sound,
  title = {Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos},
  author = {Changan Chen and Puyuan Peng and Ami Baid and Sherry Xue and Wei-Ning Hsu and David Harwath and Kristen Grauman},
  year = {2024},
  journal = {arXiv},
}
```

<!--
**Ego4DSounds/Ego4DSounds** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
