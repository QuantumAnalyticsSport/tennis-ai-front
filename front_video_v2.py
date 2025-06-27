def annotate_video_with_hits_overlay(
    video_path,
    player_info,
    player_xy,
    dico_ball,
    dico_shots,
    ellipse,
    tail,
    name,
    ball_show,
    player_annot_show,
    speed_tail,
    bounce_show,
    hit_show,
    output_path="annotated_hits_overlay.mp4",
    tail_duration=1.5): 
    import imageio.v2 as imageio

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tail_length = int(tail_duration * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_out = []  # imageio buffer

    # Step 1: Get hit classifications
    hit_results = dico_shots
    hit_overlay_text = {}
    for i, res in enumerate(hit_results):
        start = res["frame"]
        end = hit_results[i + 1]["frame"] if i + 1 < len(hit_results) else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for f in range(start, end):
            hit_overlay_text[f] = f"{res['player name']}{res['type'][13:]} at {np.round(res['speed'], 0)}km/h"

    # Step 2: Interpolated ball DataFrame
    df = pd.DataFrame.from_dict(dico_ball, orient='index')
    df.index = df.index.astype(int)
    df = df.sort_index()
    df[['x', 'y']] = df[['x', 'y']].astype('float')
    df['x_interp'] = df['x'].interpolate(limit_direction='both')
    df['y_interp'] = df['y'].interpolate(limit_direction='both')

    # Step 3: Render frames
    frame_idx = 0
    tail_buffer = deque()
    bounce_positions = []
    trajectory_buffer = {0: [], 1: []} 
    team_tracks = {0: {}, 1: {}} 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in df.index:
            row = df.loc[frame_idx]
            x, y = int(row['x_interp']), int(row['y_interp'])
            action = dico_ball.get(str(frame_idx), {}).get("action", "air")

            if ball_show:
                tail_buffer.append((x, y, frame_idx))
                while tail_buffer and frame_idx - tail_buffer[0][2] > tail_length:
                    tail_buffer.popleft()
                for j in range(1, len(tail_buffer)):
                    x1, y1, f1 = tail_buffer[j - 1]
                    x2, y2, f2 = tail_buffer[j]
                    if None in [x1, y1, x2, y2]: continue
                    dist = np.hypot(x2 - x1, y2 - y1)
                    speed = dist / max(f2 - f1, 1)
                    min_speed, max_speed = 1, 50
                    norm_speed = np.clip((speed - min_speed) / (max_speed - min_speed), 0, 1)
                    color = (
                        0,
                        int(255 * (1 - norm_speed)) if speed_tail else 255,
                        int(255 * norm_speed) if speed_tail else 255
                    )
                    cv2.line(frame, (x1, y1), (x2, y2), color, 3)
                if row.get("visible", True):
                    cv2.circle(frame, (x, y), 8, (0, 200, 255), -1)

            if bounce_show and action == "bounce":
                bounce_positions.append((x, y))
            if bounce_show:
                for bx, by in bounce_positions:
                    cv2.ellipse(frame, (bx, by), (12, 6), 0, 0, 360, (0, 255, 255), 2)

            if hit_show and action == "hit":
                cv2.circle(frame, (x, y), 20, (255, 0, 0), 2)
                cv2.line(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), 2)
                cv2.line(frame, (x - 10, y + 10), (x + 10, y - 10), (255, 0, 0), 2)

            if player_annot_show and frame_idx in hit_overlay_text:
                text = hit_overlay_text[frame_idx]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.8
                thickness = 5
                color = (30, 255, 255)
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = int(frame.shape[0] * 0.12)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Convert BGR to RGB and collect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_out.append(frame_rgb)
        frame_idx += 1

    cap.release()

    # Write with imageio
    imageio.mimsave(output_path, frames_out, fps=fps)
    print(f"✅ Video saved to: {output_path}")
