import sys
import json
import base64
import numpy as np
import librosa
import cv2
from io import BytesIO
from matplotlib import cm

def process_audio(file_path):
    """
    读取音频，计算 Mel 频谱，并准备用于 HTML 渲染的数据。
    """
    try:
        # 1. 加载音频
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    # 2. 计算 Mel Spectrogram
    # n_mels 决定了图像的高度（频率分辨率）
    # hop_length 决定了图像的宽度（时间分辨率）
    n_mels = 128
    hop_length = 512
    n_fft = 2048
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # 转为对数刻度 (dB)，这对视觉展示至关重要
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 3. 归一化到 0-255 并转为 uint8 (用于生成可视化的背景图)
    img_norm = cv2.normalize(S_db, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)
    
    # 翻转 Y 轴，因为频谱图低频在下，高频在上，但矩阵索引默认 0 在上
    img_uint8 = np.flipud(img_uint8)
    
    # 4. 生成用于显示的 Base64 图片 (使用 Matplotlib 的 colormap 上色)
    # 我们使用 'magma' 配色，因为它对比度高，看起来很像热力图
    colormap = cm.get_cmap('magma')
    im_colored = colormap(img_uint8 / 255.0) # RGBA
    im_colored = (im_colored[:, :, :3] * 255).astype(np.uint8) # RGB
    
    # 转换为 PNG 格式
    is_success, buffer = cv2.imencode(".png", im_colored)
    io_buf = BytesIO(buffer)
    b64_img = base64.b64encode(io_buf.getvalue()).decode('utf-8')

    # 5. 准备传给 JS 的原始数据
    # 为了 JS 处理方便，我们传递归一化后(0-1)的原始矩阵数据
    # 注意：需要把矩阵翻转回来或者在 JS 里处理，这里我们传翻转过的以匹配图片
    raw_data_flat = (img_uint8 / 255.0).tolist()
    
    return {
        "image_b64": b64_img,
        "spectrogram_data": raw_data_flat,
        "width": S.shape[1],
        "height": n_mels,
        "filename": file_path
    }

def generate_html(data):
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mel Spectrum & CV-based F0 Extraction</title>
    <style>
        body {{
            font-family: 'Segoe UI', sans-serif;
            background-color: #1e1e1e;
            color: #ddd;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 20px;
        }}
        h2 {{ color: #fff; margin-bottom: 10px; }}
        .controls {{
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            width: 80%;
            display: flex;
            gap: 20px;
            justify-content: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}
        label {{ font-size: 0.9em; margin-bottom: 5px; color: #aaa; }}
        input[type=range] {{ width: 200px; }}
        
        .viz-container {{
            display: flex;
            flex-direction: row;
            gap: 10px;
            overflow-x: auto;
            width: 95%;
            padding-bottom: 20px;
            background: #111;
            border: 1px solid #444;
            padding: 10px;
        }}
        
        .panel {{
            position: relative;
        }}
        
        canvas {{
            border: 1px solid #555;
            display: block;
            background-color: #000;
        }}
        
        .panel-title {{
            position: absolute;
            top: 5px;
            left: 5px;
            background: rgba(0,0,0,0.6);
            color: white;
            padding: 2px 6px;
            font-size: 12px;
            pointer-events: none;
        }}
    </style>
</head>
<body>

    <h2>CV-based F0 Extraction on Mel Spectrogram</h2>
    <div>File: {data['filename']}</div>

    <div class="controls">
        <div class="control-group">
            <label for="threshold">Energy Threshold (CV Binary Mask)</label>
            <input type="range" id="threshold" min="0" max="255" value="80">
            <span id="val-threshold">80</span>
        </div>
        <div class="control-group">
            <label for="jump-penalty">Jump Penalty (Temporal Smoothing)</label>
            <input type="range" id="jump-penalty" min="0" max="50" value="20">
            <span id="val-penalty">20</span>
        </div>
        <div class="control-group">
            <label for="look-window">Search Window (Pixels)</label>
            <input type="range" id="look-window" min="1" max="64" value="10">
            <span id="val-window">10</span>
        </div>
    </div>

    <div class="viz-container">
        <div class="panel">
            <div class="panel-title">Mel Spectrogram</div>
            <img id="mel-img" src="data:image/png;base64,{data['image_b64']}" style="height: 300px; width: auto; display: block;">
        </div>

        <div class="panel">
            <div class="panel-title">Extracted F0 Path</div>
            <canvas id="f0-canvas" height="300"></canvas>
        </div>
    </div>

    <script>
        // --- Data Ingestion ---
        // 2D Array: [Rows(Freq)][Cols(Time)] - already normalized 0-1
        // Note: Row 0 is Top (High Freq), Row N is Bottom (Low Freq)
        const specData = {json.dumps(data['spectrogram_data'])};
        const width = {data['width']};
        const height = {data['height']};
        
        const melImg = document.getElementById('mel-img');
        const canvas = document.getElementById('f0-canvas');
        const ctx = canvas.getContext('2d');

        // Set canvas dimensions to match the image
        canvas.width = width;
        canvas.height = 300; // Fixed display height
        
        // Scale factor: canvas height vs data height (n_mels)
        const scaleY = 300 / height;

        // UI References
        const sliderThresh = document.getElementById('threshold');
        const sliderPenalty = document.getElementById('jump-penalty');
        const sliderWindow = document.getElementById('look-window');
        
        const valThresh = document.getElementById('val-threshold');
        const valPenalty = document.getElementById('val-penalty');
        const valWindow = document.getElementById('val-window');

        // --- The CV Algorithm Implementation ---
        
        function runCVAlgorithm() {{
            // Parameters
            const threshold = parseInt(sliderThresh.value) / 255.0; // 0.0 to 1.0
            const maxJump = parseInt(sliderPenalty.value); // Pixel distance constraint
            const searchWindow = parseInt(sliderWindow.value);
            
            // Clear Canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // 1. Draw "Guide" (faint version of the spectrogram for reference)
            // Optional: visualization of what the CV "sees" after thresholding
            const imgData = ctx.getImageData(0, 0, width, height); // We will stretch this later
            
            // Logic: Path Tracking (Greedy with momentum/constraints)
            // We iterate through time columns (x)
            
            ctx.beginPath();
            ctx.strokeStyle = '#00ff00'; // Green line for F0
            ctx.lineWidth = 2;

            let prevY = -1; 
            let started = false;

            for (let x = 0; x < width; x++) {{
                let bestY = -1;
                let maxEnergy = -1;
                
                // Define search range (ROI - Region of Interest)
                // If we have a previous point, only search nearby pixels (CV tracking)
                let searchStart = 0;
                let searchEnd = height;

                if (prevY !== -1) {{
                    searchStart = Math.max(0, prevY - searchWindow);
                    searchEnd = Math.min(height, prevY + searchWindow);
                }}

                // Search in the column
                for (let y = searchStart; y < searchEnd; y++) {{
                    const val = specData[y][x]; // Intensity
                    
                    if (val > threshold) {{
                        // Weight the value by distance to previous to prefer continuity
                        // Cost function: Energy - Penalty * Distance
                        let score = val;
                        
                        // If we are tracking, apply distance penalty
                        if (prevY !== -1) {{
                            const dist = Math.abs(y - prevY);
                            // Simple linear penalty
                            // score -= (dist * (maxJump / 100.0)); 
                        }}
                        
                        if (score > maxEnergy) {{
                            maxEnergy = score;
                            bestY = y;
                        }}
                    }}
                }}

                // Smoothing / Momentum logic
                if (bestY !== -1) {{
                    // Determine render coordinates
                    // Data is flipped (row 0 is high freq), but let's check input
                    // Python np.flipud means Row 0 is Top (High Freq) in the array.
                    // Canvas 0 is Top. So indices match visual layout directly.
                    
                    const drawX = x;
                    const drawY = bestY * scaleY;

                    if (!started) {{
                        ctx.moveTo(drawX, drawY);
                        started = true;
                    }} else {{
                        // Apply slight smoothing for visualization
                        ctx.lineTo(drawX, drawY);
                    }}
                    
                    prevY = bestY;
                }} else {{
                    // Signal lost (silence or below threshold)
                    // Don't connect lines across large gaps of silence
                    started = false; 
                    prevY = -1; // Reset tracking
                }}
            }}
            ctx.stroke();
        }}

        // --- Event Listeners ---
        function update() {{
            valThresh.textContent = sliderThresh.value;
            valPenalty.textContent = sliderPenalty.value;
            valWindow.textContent = sliderWindow.value;
            runCVAlgorithm();
        }}

        sliderThresh.addEventListener('input', update);
        sliderPenalty.addEventListener('input', update);
        sliderWindow.addEventListener('input', update);

        // Initial Run
        // Need to wait for image to load to ensure layout is correct, 
        // but since we draw on canvas independently, we just run it.
        window.onload = function() {{
            // Ensure width matches
            // melImg.width = width; // Let CSS handle height, width scales
            update();
        }};

    </script>
</body>
</html>
    """
    return html_template

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mel_f0_viz.py <input_wav_file>")
        sys.exit(1)
        
    wav_file = sys.argv[1]
    print(f"Processing {wav_file}...")
    
    viz_data = process_audio(wav_file)
    html_content = generate_html(viz_data)
    
    output_file = "mel_f0_output.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Done! Open '{output_file}' in your browser.")
