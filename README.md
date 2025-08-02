# AutoCommentator: AI Football Match Commentary Generator

This project is a prototype system for automatic generation of football match commentary using computer vision, natural language generation, and speech synthesis. It analyzes football match videos, detects key events, and produces realistic audio commentary in sync with the gameplay.

## Features

- **Entity Detection**: Identifies players, referees, goalkeepers, and the ball using YOLOv8 object detection models.
---<img width="950" height="543" alt="detectare jucatori" src="https://github.com/user-attachments/assets/5e632afa-a004-4546-9b71-7af6ebf5c098" />
  
- **Field Landmark Detection**: Uses YOLOv8 Pose to detect keypoints on the football field (e.g. corners, center circle, penalty area edges), enabling accurate spatial mapping via homography.
![pitch_detection](https://github.com/user-attachments/assets/1b8a05ae-b9f6-4e2f-94f9-52d1e08f57ac)
  
- **Object Tracking**: Integrates the ByteTrack algorithm to maintain consistent player identities across video frames.
- **Team Classification**: Applies SigLIP (a vision-language model) to extract visual embeddings of player crops. These embeddings are projected to a lower-dimensional space using UMAP and clustered with KMeans to assign players to teams in an unsupervised manner.
<img width="950" height="543" alt="echipe" src="https://github.com/user-attachments/assets/81f335c3-9cf8-4e9d-b5e2-470063a70323" />

  
- **Event Analysis**: Estimates ball possession and detects key gameplay events such as passes, interceptions, and tackles based on player positions and transitions. Approximate shot detection is inferred from high-speed movement toward the goal.
- **Commentary Generation**: Generates context-aware natural language descriptions for each event using OpenAI’s GPT model, with prompts including spatial positions and time constraints.
- **Speech Synthesis**: Transforms generated text into natural-sounding audio using ElevenLabs, with customizable voice and language options.
- **Web Application**: Provides a simple Flask-based interface for video upload and playback of processed clips with overlaid visuals and synchronized audio commentary.
