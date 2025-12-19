import sys
import os
import base64
import io
import numpy as np
import librosa
from PIL import Image
import matplotlib.cm as cm

def process_audio_to_base64(audio_path):
    """
    读取 WAV，计算 Mel 频谱，归一化并转为 Base64 图片字符串。
    """
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        sys.exit(1)

    print(f"Processing audio: {audio_path}...")
    
    # 1. 加载音频
    y, sr = librosa.load(audio_path)
    
    # 2. 计算 Mel Spectrogram
    # n_mels 决定了纵向的分辨率
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # 3. 转为对数刻度 (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 4. 归一化到 0-255 并转为 uint8 (为了生成图片)
    # 翻转 Y 轴，因为 librosa 默认低频在下，但在数组中是 index 0
    S_dB = np.flipud(S_dB) 
    
    min_val = S_dB.min()
    max_val = S_dB.max()
    norm_S = 255 * (S_dB - min_val) / (max_val - min_val)
    norm_S = norm_S.astype(np.uint8)
    
    # 5. 应用颜色映射 (Colormap) - 使用 'magma' 或 'inferno' 效果较好
    # Matplotlib 的 colormap 返回 0-1 的 float，我们需要转回 0-255
    cmap = cm.get_cmap('magma')
    im_colored = cmap(norm_S / 255.0) 
    im_uint8 = (im_colored * 255).astype(np.uint8)
    
    # 6. 转为 PIL Image 并保存到内存 Buffer
    img = Image.fromarray(im_uint8)
    
    # 这里不需要坐标轴，直接是纯像素
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def generate_html(base64_img, output_filename):
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Mel Spectrum Ridge Detection</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1e1e1e;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h2 {{ margin-bottom: 10px; color: #fff; }}
        .controls {{
            background: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}
        label {{ font-size: 0.9em; margin-bottom: 5px; color: #aaa; }}
        input[type=range] {{ width: 200px; cursor: pointer; }}
        
        .container {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }}
        .panel {{
            background: #000;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #444;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .panel h3 {{ margin: 0 0 10px 0; font-size: 1em; color: #888; }}
        canvas {{
            max-width: 100%;
            height: auto;
            image-rendering: pixelated; /* 保持像素清晰，适合频谱图 */
        }}
    </style>
</head>
<body>

    <h2>Mel Spectrum Ridge Detector</h2>

    <div class="controls">
        <div class="control-group">
            <label for="threshold">亮度阈值 (Noise Gate): <span id="val-thresh">30</span></label>
            <input type="range" id="threshold" min="0" max="255" value="30">
        </div>
        <div class="control-group">
            <label for="ridgeObj">脊线灵敏度 (Vertical Contrast): <span id="val-ridge">15</span></label>
            <input type="range" id="ridgeObj" min="0" max="100" value="15">
        </div>
        <div class="control-group">
            <label for="gain">显示增益 (Output Gain): <span id="val-gain">2.0</span></label>
            <input type="range" id="gain" min="0.5" max="5.0" step="0.1" value="2.0">
        </div>
    </div>

    <div class="container">
        <div class="panel">
            <h3>Original Mel Spectrum</h3>
            <canvas id="canvasSource"></canvas>
        </div>
        <div class="panel">
            <h3>Extracted Horizontal Ridges</h3>
            <canvas id="canvasDest"></canvas>
        </div>
    </div>

    <img id="sourceImg" src="data:image/png;base64,{base64_img}" style="display:none;" />

    <script>
        window.onload = function() {{
            const img = document.getElementById('sourceImg');
            const cSrc = document.getElementById('canvasSource');
            const ctxSrc = cSrc.getContext('2d');
            const cDest = document.getElementById('canvasDest');
            const ctxDest = cDest.getContext('2d');

            // Controls
            const sThreshold = document.getElementById('threshold');
            const sRidge = document.getElementById('ridgeObj');
            const sGain = document.getElementById('gain');

            // Display value updaters
            const updateLabels = () => {{
                document.getElementById('val-thresh').innerText = sThreshold.value;
                document.getElementById('val-ridge').innerText = sRidge.value;
                document.getElementById('val-gain').innerText = sGain.value;
            }};

            // Main Processing Function
            function process() {{
                const w = img.width;
                const h = img.height;

                // Sync canvas sizes
                if (cSrc.width !== w) {{ cSrc.width = w; cSrc.height = h; }}
                if (cDest.width !== w) {{ cDest.width = w; cDest.height = h; }}

                // Draw source
                ctxSrc.drawImage(img, 0, 0);
                
                // Get pixel data
                const srcData = ctxSrc.getImageData(0, 0, w, h);
                const dstData = ctxDest.createImageData(w, h);
                
                const sBuf = srcData.data;
                const dBuf = dstData.data;

                const thresh = parseInt(sThreshold.value);
                const ridgeSens = parseInt(sRidge.value);
                const gain = parseFloat(sGain.value);

                // CV Logic: Horizontal Ridge Detection
                // 我们遍历每个像素，检查它在垂直方向上是否是局部最大值（脊）
                // 并且亮度是否超过阈值
                
                for (let y = 1; y < h - 1; y++) {{
                    for (let x = 0; x < w; x++) {{
                        const idx = (y * w + x) * 4;
                        
                        // 获取当前像素及上下像素的亮度 (简单取 R 通道，因为是灰度或伪彩色)
                        // 为了准确性，我们计算简单的平均亮度 (R+G+B)/3
                        const val = (sBuf[idx] + sBuf[idx+1] + sBuf[idx+2]) / 3;
                        
                        // 上方像素
                        const idxUp = ((y - 1) * w + x) * 4;
                        const valUp = (sBuf[idxUp] + sBuf[idxUp+1] + sBuf[idxUp+2]) / 3;
                        
                        // 下方像素
                        const idxDown = ((y + 1) * w + x) * 4;
                        const valDown = (sBuf[idxDown] + sBuf[idxDown+1] + sBuf[idxDown+2]) / 3;

                        // 核心算法：
                        // 1. 自身亮度必须大于阈值 (Noise Gate)
                        // 2. 自身亮度必须显著高于上方像素 (Ridge Top)
                        // 3. 自身亮度必须显著高于下方像素 (Ridge Bottom)
                        // 这构成了一个垂直方向的"山峰"，即横向延伸的脊线
                        
                        let isRidge = false;
                        if (val > thresh) {{
                            if ((val > valUp + ridgeSens) && (val > valDown + ridgeSens)) {{
                                isRidge = true;
                            }}
                        }}

                        if (isRidge) {{
                            // 设为白色 (或者根据原始强度 * gain)
                            let outVal = val * gain;
                            if (outVal > 255) outVal = 255;
                            
                            dBuf[idx] = outVal;     // R
                            dBuf[idx+1] = outVal;   // G
                            dBuf[idx+2] = outVal;   // B
                            dBuf[idx+3] = 255;      // Alpha
                        }} else {{
                            // 背景全黑
                            dBuf[idx] = 0;
                            dBuf[idx+1] = 0;
                            dBuf[idx+2] = 0;
                            dBuf[idx+3] = 255;
                        }}
                    }}
                }}

                ctxDest.putImageData(dstData, 0, 0);
            }}

            // Event Listeners
            sThreshold.oninput = () => {{ updateLabels(); process(); }};
            sRidge.oninput = () => {{ updateLabels(); process(); }};
            sGain.oninput = () => {{ updateLabels(); process(); }};

            // Initial Run
            updateLabels();
            process();
        }};
    </script>
</body>
</html>
    """
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Success! HTML generated: {output_filename}")

if __name__ == "__main__":
    # 默认处理 sys.argv[1]，如果没有参数则提示
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_wav_file>")
        # 为了演示方便，如果用户没传参数，可以尝试读取当前目录下的 demo.wav (可选)
        sys.exit(1)
    
    input_wav = sys.argv[1]
    output_html = os.path.splitext(input_wav)[0] + "_spectrum.html"
    
    # 1. 音频 -> 图片 Base64
    img_data = process_audio_to_base64(input_wav)
    
    # 2. 生成带算法的 HTML
    generate_html(img_data, output_html)
