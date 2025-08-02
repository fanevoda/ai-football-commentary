import os
import sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import re
from openai import OpenAI
import json
from elevenlabs.client import ElevenLabs
from elevenlabs import save

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip  import CompositeAudioClip


import numpy as np
import supervision as sv
def embed_crops(crops, embedding_model, processor, device, batch_size=32):

    crops = [sv.cv2_to_pillow(crop) for crop in crops]
    batches = chunked(crops, batch_size)
    data = []

    with torch.no_grad():
        for batch in tqdm(batches, desc='embedding extraction'):
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = embedding_model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
            data.append(embeddings)

    return np.concatenate(data)

def collect_player_crops(video_path, model, player_id=2, stride=30):
    import supervision as sv
    from tqdm import tqdm

    frame_generator = sv.get_video_frames_generator(video_path, stride=stride)
    crops = []

    for frame in tqdm(frame_generator, desc="collecting crops"):
        result = model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == player_id]
        crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

    return crops

def cluster_embeddings(embeddings, n_clusters=2):
    import umap
    from sklearn.cluster import KMeans

    reducer = umap.UMAP(n_components=3)
    reduced = reducer.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)

    return labels, reduced


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)



import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from collections import Counter

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from more_itertools import chunked
from collections import Counter
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

# Global Pitch Configuration
CONFIG = SoccerPitchConfiguration()
CONFIG.pitch_color = '#333333'
CONFIG.pitch_alpha = 0.6


def embed_crops(crops, embedding_model, processor, device, batch_size=32):
    """
    Generate embeddings for a list of image crops.
    """
    crops_pil = [sv.cv2_to_pillow(crop) for crop in crops]
    batches = chunked(crops_pil, batch_size)
    embeddings_list = []

    with torch.no_grad():
        for batch in tqdm(batches, desc='Embedding extraction'):
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = embedding_model(**inputs)
            # Use pooled output if available, otherwise mean of last hidden state
            if hasattr(outputs, 'pooler_output'):
                emb = outputs.pooler_output.cpu().numpy()
            else:
                emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings_list.append(emb)

    return np.concatenate(embeddings_list, axis=0)


def collect_player_crops(video_path, detection_model, player_id=2, stride=15):
    """
    Extract player crops from a video using a detection model.
    """
    crops = []
    frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    for frame in tqdm(frame_gen, desc='Collecting player crops'):
        result = detection_model.infer(frame, confidence=0.3)[0]
        dets = sv.Detections.from_inference(result)
        dets = dets.with_nms(threshold=0.5, class_agnostic=True)
        dets = dets[dets.class_id == player_id]
        crops.extend([sv.crop_image(frame, box) for box in dets.xyxy])
    return crops


def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """
    Assign goalkeeper detections to the nearest team based on player centroids.
    """
    gk_coords = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    player_coords = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    # Compute centroids
    centroids = []
    for team_id in [0, 1]:
        pts = player_coords[players.class_id == team_id]
        centroids.append(pts.mean(axis=0) if len(pts) else np.array([0, 0]))
    # Assign each GK to closest centroid
    gk_ids = []
    for pt in gk_coords:
        d0 = np.linalg.norm(pt - centroids[0])
        d1 = np.linalg.norm(pt - centroids[1])
        gk_ids.append(0 if d0 < d1 else 1)
    return np.array(gk_ids, dtype=int)


def make_vid(SOURCE_VIDEO_PATH, debug = False):

    import cv2
    from tqdm import tqdm
    import warnings
    import supervision as sv
    from collections import Counter
    from sports.common.view import ViewTransformer
    from video_processor.model_loader import load_detection_model, load_field_detection_model, load_team_classifier

    PLAYER_DETECTION_MODEL = load_detection_model()
    FIELD_DETECTION_MODEL = load_field_detection_model()
    team_classifier = load_team_classifier()
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    OUTPUT_VIDEO_PATH = "outputs/output_video.mp4"

    # -------------------- Setup --------------------
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    poss_candidate_history = []
    last_stable_possessor = None

    event_overlay_counter = 0
    possession_event_label = ""


    last_ball_pos = None

    untrusted_ball_counter = 0

    structured_events = []

    # ByteTrack & Annotators
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    tracker = sv.ByteTrack()
    tracker.reset()

    TEAM_1_COLOR = '#BB8FCE'  # mov
    TEAM_2_COLOR = '#5DADE2'  # albastru
    REFEREE_COLOR = '#F4D03F'  # galben
    BALL_COLOR = sv.Color.WHITE
    BALL_EDGE = sv.Color.BLACK

    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex([TEAM_1_COLOR, TEAM_2_COLOR, REFEREE_COLOR]),
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex([TEAM_1_COLOR, TEAM_2_COLOR, REFEREE_COLOR]),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )

    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex(REFEREE_COLOR),
        base=20, height=17
    )

    # -------------------- Main Loop --------------------
    for i, frame in enumerate(tqdm(frame_generator, desc="Processing video")):
        if i == 200:
            break

        # --- Detection & Tracking ---
        detections = sv.Detections.from_inference(PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0])
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = tracker.update_with_detections(all_detections.with_nms(threshold=0.5, class_agnostic=True))

        # --- Classify Players & Merge Detections ---
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)
        goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
        referees_detections.class_id -= 1

        all_detections = sv.Detections.merge([players_detections, goalkeepers_detections, referees_detections])
        all_detections.class_id = all_detections.class_id.astype(int)
        labels = [f"#{tracker_id}" for tracker_id in all_detections.tracker_id]

        # --- Field Transformation ---
        key_points = sv.KeyPoints.from_inference(FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0])
        filter = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        if len(frame_reference_points) < 4:
            writer.write(frame)
            continue

        transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
        pitch_goalkeepers_xy = transformer.transform_points(
            goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        pitch_players_xy = transformer.transform_points(
            players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        pitch_ball_xy = transformer.transform_points(ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
        pitch_referees_xy = transformer.transform_points(
            referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))

        # --- no penalty ball ---

        MAX_BALL_JUMP = 500  # adjust based on your pitch size

        if pitch_ball_xy.shape[0] > 0:
            detected_ball = pitch_ball_xy[0]

            if last_ball_pos is not None:
                jump = np.linalg.norm(detected_ball - last_ball_pos)

                if jump > MAX_BALL_JUMP:
                    untrusted_ball_counter += 1
                    print(f"⚠️ Unrealistic jump ({jump:.1f}) → using last_ball_pos [{untrusted_ball_counter=}]")
                    current_ball = last_ball_pos
                else:
                    untrusted_ball_counter = max(untrusted_ball_counter - 1, 0)
                    current_ball = detected_ball
                    last_ball_pos = detected_ball
            else:
                current_ball = detected_ball
                last_ball_pos = detected_ball

            #  Safe to trust new detection if fallback has been used too long
            if untrusted_ball_counter > 5:
                print(" Forcing re-trust of new ball after prolonged untrust.")
                current_ball = detected_ball
                last_ball_pos = detected_ball
                untrusted_ball_counter = 0

        else:
            untrusted_ball_counter += 1
            print(f"Ball not detected → fallback to last_ball_pos [{untrusted_ball_counter=}]")
            current_ball = last_ball_pos

        # --- Ball Possession ---
        possessor_id, possessor_team, min_dist = None, None, float('inf')
        if current_ball is not None:
            for idx, player_pos in enumerate(pitch_players_xy):
                dist = np.linalg.norm(player_pos - current_ball)
                if dist < 300 and dist < min_dist:
                    min_dist = dist
                    possessor_id = players_detections.tracker_id[idx]
                    possessor_team = players_detections.class_id[idx]

        # Record current possession
        # Update possession candidates
        if possessor_id is not None:
            print(poss_candidate_history)

            # Find player's pitch position
            player_xy = None

            for idx, pid in enumerate(players_detections.tracker_id):
                if pid == possessor_id:
                    player_xy = pitch_players_xy[idx]
                    break

            if player_xy is not None:
                poss_candidate_history.append([possessor_id, possessor_team, player_xy])

            # Maintain sliding window
            if len(poss_candidate_history) > 10:
                poss_candidate_history.pop(0)

            # Use tuple for counting
            # Count just (id, team) for voting
            most_common = Counter(
                (p[0], p[1]) for p in poss_candidate_history
            ).most_common(1)

            # Decide stable possessor
            if most_common and most_common[0][1] >= 6:
                stable_possessor = most_common[0][0]  # (id, team)
            else:
                stable_possessor = None

            # Event classification
            if stable_possessor != last_stable_possessor and stable_possessor is not None:
                event_type = None
                event_message = ""

                prev_id, prev_team = last_stable_possessor if last_stable_possessor else (None, None)
                curr_id, curr_team = stable_possessor

                prev_player_xy = None
                for p in reversed(poss_candidate_history):
                    if p[0] == prev_id:
                        prev_player_xy = p[2]
                        break

                if prev_team == curr_team:
                    event_type = "PASS"
                    event_message = f"PASS: #{prev_id} → #{curr_id} (Team {curr_team})"
                elif prev_player_xy is not None and np.linalg.norm(prev_player_xy - current_ball) > 300:
                    event_type = "INTERCEPTION"
                    event_message = f"INTERCEPTION: #{prev_id} lost to #{curr_id} (opponent)"
                else:
                    event_type = "TACKLE"
                    event_message = f"TACKLE: #{prev_id} tackled by #{curr_id} (opponent)"

                last_stable_possessor = stable_possessor
                possession_event_label = event_message
                event_overlay_counter = 45  # show for 1.5 seconds

                structured_events.append({
                    "frame": i,
                    "time_sec": i / fps,
                    "event_type": event_type,
                    "description": event_message,
                    "ball_position": [int(x) for x in current_ball.tolist()] if current_ball is not None else None,
                    "possessor": {
                        "id": int(curr_id),
                        "team": int(curr_team),
                        "position": [int(x) for x in player_xy.tolist()] if player_xy is not None else None
                    },
                    "previous_possessor": {
                        "id": int(prev_id) if prev_id is not None else None,
                        "team": int(prev_team) if prev_team is not None else None,
                        "position": [int(x) for x in prev_player_xy] if prev_player_xy is not None else None
                    },
                    "player_positions": [
                        {
                            "id": int(players_detections.tracker_id[j]),
                            "team": int(players_detections.class_id[j]),
                            "position": [int(x) for x in pitch_players_xy[j]]
                        }
                        for j in range(len(players_detections))
                    ]
                })

            if current_ball is not None and last_stable_possessor is not None:
                curr_ball = current_ball
                poss_id, poss_team = last_stable_possessor

                for idx, gk_xy in enumerate(pitch_goalkeepers_xy):
                    gk_team = goalkeepers_detections.class_id[idx]

                    if gk_team != poss_team:
                        distance_to_gk = np.linalg.norm(curr_ball - gk_xy)

                        if distance_to_gk < 400:  # adjust threshold as needed
                            goal_label = f"GOAL ATTEMPT: #{poss_id} (Team {poss_team}) → GK (Team {gk_team})"
                            print(goal_label)
                            goal_overlay_counter = 45
                            break

        # --- Radar Drawing ---

        # SCALEA IAIC ##########################################################################3333

        radar_frame = draw_pitch(CONFIG, sv.Color(50, 50, 50), sv.Color.WHITE, padding=30, line_thickness=2)
        radar_frame = draw_points_on_pitch(CONFIG, pitch_goalkeepers_xy[goalkeepers_detections.class_id == 0],
                                           sv.Color.from_hex('00BFFF'), sv.Color.BLACK, 20, pitch=radar_frame)
        radar_frame = draw_points_on_pitch(CONFIG, pitch_goalkeepers_xy[goalkeepers_detections.class_id == 1],
                                           sv.Color.from_hex('FF1493'), sv.Color.BLACK, 20, pitch=radar_frame)

        radar_frame = draw_points_on_pitch(CONFIG, pitch_ball_xy, sv.Color.WHITE, sv.Color.BLACK, 17, pitch=radar_frame)
        radar_frame = draw_points_on_pitch(CONFIG, pitch_players_xy[players_detections.class_id == 0],
                                           sv.Color.from_hex('00BFFF'), sv.Color.BLACK, 25, pitch=radar_frame)
        radar_frame = draw_points_on_pitch(CONFIG, pitch_players_xy[players_detections.class_id == 1],
                                           sv.Color.from_hex('FF1493'), sv.Color.BLACK, 25, pitch=radar_frame)
        radar_frame = draw_points_on_pitch(CONFIG, pitch_referees_xy, sv.Color.from_hex('FFD700'), sv.Color.BLACK, 16,
                                           pitch=radar_frame)

        # --- Frame Annotations ---
        annotated = frame.copy()

        if debug:
            annotated = ellipse_annotator.annotate(scene=annotated, detections=all_detections)
            annotated = label_annotator.annotate(scene=annotated, detections=all_detections, labels=labels)
            annotated = triangle_annotator.annotate(scene=annotated, detections=ball_detections)

            if possessor_id is not None:
                index = list(players_detections.tracker_id).index(possessor_id)
                annotated = sv.EllipseAnnotator(color=sv.Color(0, 255, 0), thickness=3).annotate(
                    scene=annotated,
                    detections=players_detections[index:index + 1]
                )

        # sv.plot_image(radar_frame)

        # --- Radar Overlay ---
        radar_resized = cv2.resize(radar_frame, (int(width * 0.2), int(height * 0.2)))
        src_pts = np.float32([[0, 0], [radar_resized.shape[1], 0], [radar_resized.shape[1], radar_resized.shape[0]],
                              [0, radar_resized.shape[0]]])
        dst_pts = np.float32([
            [radar_resized.shape[1] * 0.2, 0],
            [radar_resized.shape[1] * 0.8, 0],
            [radar_resized.shape[1], radar_resized.shape[0] * 0.6],
            [0, radar_resized.shape[0] * 0.6]
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        radar_warped = cv2.warpPerspective(radar_resized, M, (radar_resized.shape[1], radar_resized.shape[0]))

        x_offset = (width - radar_warped.shape[1]) // 2
        y_offset = height - radar_warped.shape[0] - 20
        mask = cv2.cvtColor(radar_warped, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        roi = annotated[y_offset:y_offset + radar_warped.shape[0], x_offset:x_offset + radar_warped.shape[1]]
        fg = cv2.bitwise_and(radar_warped, radar_warped, mask=binary_mask)
        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(binary_mask))
        blended = cv2.add(bg, fg)
        annotated[y_offset:y_offset + radar_warped.shape[0], x_offset:x_offset + radar_warped.shape[1]] = blended

        # --- Possession Text ---
        if possessor_id is not None:
            team_str = f"Team {possessor_team}" if possessor_team in [0, 1] else "Unknown Team"
            text = f"Possession: #{possessor_id} ({team_str}) | dist={min_dist:.2f}"
            cv2.putText(annotated, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # print(pitch_players_xy)
        # print(pitch_ball_xy)

        # --- debug pase ---

        # --- Draw Possession Frequency on Frame ---
        x_pos = 50
        y_pos = 200
        line_spacing = 30

        cv2.putText(
            annotated,
            "Possession Frequency (last 10):",
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # white
            2,
            cv2.LINE_AA
        )

        # Iterate and draw each player's count
        for i, ((player_id, team_id), count) in enumerate(
                Counter((p[0], p[1]) for p in poss_candidate_history).items()
        ):
            line = f"Player #{player_id} (Team {team_id}): {count} frames"
            cv2.putText(
                annotated,
                line,
                (x_pos, y_pos + (i + 1) * line_spacing),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,  # increased font scale
                (200, 200, 0),  # light yellow
                2,  # increased thickness
                cv2.LINE_AA
            )

        # --- Draw event message if active ---
        if event_overlay_counter > 0 and possession_event_label:
            cv2.putText(
                annotated,
                f"{possession_event_label}",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),  # yellow
                2,
                cv2.LINE_AA
            )
            event_overlay_counter -= 1

        print(pitch_ball_xy)

        writer.write(annotated)

    print("Structured Events:")
    for event in structured_events:
        print(event)

    return structured_events

    writer.release()


def add_timers(events):
    for i, event in enumerate(events):
        start_time = event['time_sec']
        if i < len(events) - 1:
            end_time = events[i + 1]['time_sec']
            duration = end_time - start_time
        else:
            duration = 3.0  # default duration for last event
        events[i]['occurence_time'] = duration
        print(f"{start_time:.2f}s (+{duration:.2f}s) - "
              f"{event['event_type']}: {event['description']}")

        # --- Team orientation logic ---

        players = event.get("player_positions", [])

        team_0_x = [p["position"][0] for p in players if p["team"] == 0]
        team_1_x = [p["position"][0] for p in players if p["team"] == 1]

        if team_0_x and team_1_x:
            avg_0 = sum(team_0_x) / len(team_0_x)
            avg_1 = sum(team_1_x) / len(team_1_x)

            if avg_0 < avg_1:
                event["team_left"] = 0
                event["team_right"] = 1
            else:
                event["team_left"] = 1
                event["team_right"] = 0
        else:
            # Fallback if data is incomplete
            event["team_left"] = None
            event["team_right"] = None

        print(f"{start_time:.2f}s (+{duration:.2f}s) - "
              f"{event['event_type']}: {event['description']}")
    return events


def extract_commentary_lines(raw_text):
    return re.findall(r'"([^"]+)"', raw_text)


def getCommentary(events, response_language = "english"):

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    assistant = client.beta.assistants.create(
        name="Football Commentator",
         instructions="""
You are a sports commentator AI generating short, vivid football commentary based on in-game events.

Pitch details:
- Pitch size: 12000 (length) x 7000 (width)
- Positions are in (x, y) format:
  - x = 0 is the far **left** of the pitch
  - x = 12000 is the far **right**
  - y = 0 is the **bottom** touchline
  - y = 7000 is the **top** touchline

Use positional awareness to infer direction of play (e.g., a team attacking from left to right).

Player roles:
- Use terms like **centerback**, **fullback**, **midfielder**, **winger**, or **striker** based on position.
- Never refer to players by number.

Commentary Guidelines:
- Focus on the `event_type`, ball movement, and tactical implications.
- Incorporate field direction (left-to-right/right-to-left) if it helps.
- Never mention "Team 0" or "Team 1" explicitly — just describe play naturally.
- Use vivid verbs: surges, intercepts, blasts, threads, etc.
- Always reply in a **single sentence**, respecting word count limits.
- Imagine you're narrating a fast-paced highlight reel.
""",

        model="gpt-4"
    )

    commentary_lines = []

    total = 0

    for i, event in enumerate(events):
        thread = client.beta.threads.create()

        print(event)

        duration = event.get("occurence_time", 3.0)

        word_limit = int(duration)

        print(duration)

        team_left = event.get("team_left")
        team_right = event.get("team_right")

        orientation_text = ""
        if team_left is not None and team_right is not None:
            orientation_text = (
                f"Team {team_left} is positioned on the **left side** of the pitch,\n"
                f"while Team {team_right} is on the **right**.\n"
                "This can inform direction of play, attacking sides, and transitions.\n"
            )

        content = (
            f"EVENT = {json.dumps(event)}\n\n"
            f"The event duration is {duration:.2f} seconds.\n"
            f"{orientation_text}"
            f"Generate ONE short football commentary line, strictly limited to **{word_limit} words or fewer**.\n"
            "Focus on the `event_type`, basic tactical implication, and directional flow if possible.\n"
            "Avoid player numbers, timestamps, or extra explanation. Return only the commentary string.\n"
            f"Please reply in {response_language}."
        )

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=content
        )

        run = client.beta.threads.runs.create_and_poll(
            assistant_id=assistant.id,
            thread_id=thread.id
        )

        print("→ run status:", run.status, "run:", run)
        msgs = client.beta.threads.messages.list(thread_id=thread.id).data
        print("→ all messages after run:", [(m.role, m.content) for m in msgs])

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        for message in reversed(messages.data):
            if message.role == "assistant":
                commentary = message.content[0].text.value.strip().strip('"')
                commentary_lines.append(commentary)
                print(f"{i + 1}. {commentary}")
                break

    return commentary_lines


def generate_commentary_audio(
    commentary_lines,
    voice_ids=None,
    model_id="eleven_multilingual_v2",
    output_dir="commentary_audio"
):

    if isinstance(commentary_lines, str):
        commentary_lines = json.loads(commentary_lines)

    # Default voices if none provided
    if voice_ids is None:
        voice_ids = ["EXAVITQu4vr4xnSDxMaL", "ErXwobaYiN019PkySvjV"]  # Rachel, Adam

    # Authenticate
    client = ElevenLabs(api_key='sk_7545a47ab7e1bab1ef52a96e37b26ef0193ce8e4519d2a03')

    # Prepare output folder
    os.makedirs(output_dir, exist_ok=True)

    audio_paths = []
    for i, line in enumerate(commentary_lines):
        voice_id = voice_ids[i % len(voice_ids)]
        audio = client.text_to_speech.convert(
            text=line,
            voice_id=voice_id,
            model_id=model_id
        )
        path = os.path.join(output_dir, f"line_{i:02}.mp3")
        save(audio, path)
        audio_paths.append(path)

    print(f"✅ Generated {len(audio_paths)} audio clips in “{output_dir}”")
    return audio_paths

def add_audio(audio_paths, event_times, OUTPUT_VIDEO_PATH):
    # Load the base video
    video = VideoFileClip(OUTPUT_VIDEO_PATH)

    # Prepare audio clips with truncation if overlapping
    audio_clips = []
    for i, (path, start_time) in enumerate(zip(audio_paths, event_times)):
        audio = AudioFileClip(path)

        # Determine max allowed duration
        if i < len(event_times) - 1:
            max_end_time = event_times[i + 1]
            duration = max_end_time - start_time
            if duration > 0:
                audio = audio.subclipped(0, min(audio.duration, duration))
        # No truncation for the last clip
        audio = audio.with_start(start_time)
        audio_clips.append(audio)

    # Combine into one audio track
    composite_audio = CompositeAudioClip(audio_clips)

    # Set the composite audio to the video
    final_video = video.with_audio(composite_audio)

    # Export
    final_video.write_videofile("commentated_video.mp4", codec="libx264", audio_codec="aac", preset="ultrafast")




def merge_events(events):
    if not events:
        return {}

    print(events)

    # Merge descriptions with separator
    combined_description = " | ".join(event.get("description", "") for event in events)
    print(combined_description)

    # Sum total occurrence time
    total_duration = sum(event.get("occurence_time", 0) for event in events)

    # Build merged event dictionary
    merged_event = {
        "frame": events[0].get("frame"),
        "time_sec": events[0].get("time_sec"),
        "event_type": "GROUPED",
        "description": combined_description,
        "occurence_time": total_duration,
        "ball_position": events[-1].get("ball_position"),
        "possessor": events[-1].get("possessor"),
        "previous_possessor": events[0].get("previous_possessor")
    }

    return merged_event

def group_short_events(events, min_duration=3, max_group_size=5):
    grouped = []
    buffer = []

    for event in events:
        duration = event.get('occurence_time', 0)

        if duration < min_duration:
            buffer.append(event)
            if len(buffer) >= max_group_size:
                grouped.append(merge_events(buffer))
                buffer = []
        else:
            if buffer:
                grouped.append(merge_events(buffer))
                buffer = []
            grouped.append(event)

    if buffer:
        grouped.append(merge_events(buffer))

    return grouped




def generate_commentated_video(input_path, output_path, params=None):
    try:
        print("[INFO] Training team classifier...")
        subprocess.run(
            [sys.executable, os.path.join("scripts", "train_team_classifier.py"), "--video", input_path],
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to train team classifier: {e}")
        raise RuntimeError("Team classifier training failed") from e


    language = "english"
    voice1 = "rachel"
    voice2 = "adam"
    debug = False
    if params:
        if "language" in params:
            language = params["language"]
        if "voice1" in params:
            voice1 = params["voice1"]
        if "voice2" in params:
            voice2 = params["voice2"]
        debug = params.get("debug", False)
        if isinstance(debug, str):
            debug = debug.lower() in ["true", "1", "yes"]

    VOICE_MAP = {
        "bogdan": "5asM3ZxsegvXfXI5vqKQ",
        "andrei": "ANRS3e9rxJEXUpOhaPDb",
        "corina": "RjgBjNgGkuZd49zyCxIq",

        "eva": "RgXx32WYOGrd7gFNifSf",
        "martin" : "Wl3O9lmFSMgGFTTwuS6f",

        "adam": "EXAVITQu4vr4xnSDxMaL",
        "rachel": "ErXwobaYiN019PkySvjV"
    }

    voice_id_1 = VOICE_MAP.get(voice1.lower(), VOICE_MAP)
    voice_id_2 = VOICE_MAP.get(voice2.lower(), VOICE_MAP)
    voice_ids = [voice_id_1, voice_id_2]


    structured_events = make_vid(input_path, debug)
    structured_events.pop(0)


    structured_events = add_timers(structured_events)
    print(structured_events)
    structured_events = group_short_events(structured_events)
    print(structured_events)

    commentary_lines = getCommentary(events=structured_events, response_language=language)

    audio_paths = generate_commentary_audio(commentary_lines, voice_ids)

    event_times = [round(event['time_sec'], 2) for event in structured_events]

    add_audio(audio_paths, event_times, "outputs/output_video.mp4")

    os.rename("commentated_video.mp4", output_path)




if __name__ == "__main__":
    from pprint import pprint


    # Sample test data
    test_events = [
        {
            "frame": 10,
            "time_sec": 1.2,
            "event_type": "TACKLE",
            "description": "Player A tackled Player B",
            "occurence_time": 0.5,
            "ball_position": [1000, 2000],
            "possessor": {"id": 5, "team": 0, "position": [1000, 2000]},
            "previous_possessor": {"id": 7, "team": 1, "position": [1100, 2100]},
            "player_positions": [],
            "team_left": 0,
            "team_right": 1
        },
        {
            "frame": 20,
            "time_sec": 2.1,
            "event_type": "PASS",
            "description": "Player A passed to Player C",
            "occurence_time": 0.8,
            "ball_position": [1200, 2300],
            "possessor": {"id": 8, "team": 0, "position": [1200, 2300]},
            "previous_possessor": {"id": 5, "team": 0, "position": [1000, 2000]},
            "player_positions": [],
            "team_left": 0,
            "team_right": 1
        }
    ]

    # Run test
    merged = merge_events(test_events)

    print("✅ Merged Event:")
    pprint(merged)

