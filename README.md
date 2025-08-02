# 🎙️ AutoCommentator: AI Football Match Commentary Generator

This project is a prototype system for **automatic generation of football match commentary** using computer vision, natural language generation, and speech synthesis. It analyzes football videos, detects key events, and produces realistic audio commentary in real time.

---

## Features

-  **Entity Detection**: Identifies players, referees, goalkeepers, and the ball using YOLOv8.
-  **Object Tracking**: Uses ByteTrack to track players across frames.
-  **Team Classification**: Unsupervised visual clustering to separate players into teams.
-  **Event Analysis**: Detects passes, interceptions, tackles, and estimated shots.
-  **Commentary Generation**: Uses OpenAI’s GPT API to describe the events.
-  **Speech Synthesis**: Converts text into realistic audio via ElevenLabs API.
-  **Web App**: Flask-based frontend to upload videos and get the augmented result.
