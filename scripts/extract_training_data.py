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