import sys
import os
import base64
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 设置 Matplotlib 后端为非交互式
plt.switch_backend('Agg')

def create_mel_html(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    print(f"Processing {audio_path}...")

    # 1. 加载音频
    try:
        y, sr = librosa.load(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # 2. 计算 Mel 频谱
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 3. 绘制图像 (无坐标轴)
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    librosa.display.specshow(S_dB, sr=sr, fmax=8000, cmap='magma')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
    buf.seek(0)
    
    # 4. 转为 Base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    img_src = f"data:image/png;base64,{img_base64}"

    # 5. HTML 模板 (使用普通字符串，避免 f-string 与 JS 大括号冲突)
    # 我们使用一个唯一的标记 {{IMG_DATA}} 来后续替换
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mel Spectrogram Canny Edge Detection</title>
    <style>
        body { font-family: sans-serif; background: #1e1e1e; color: #fff; padding: 20px; text-align: center; }
        .container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 20px; }
        .canvas-wrapper { position: relative; border: 1px solid #444; }
        canvas { display: block; max-width: 100%; height: auto; }
        .label { position: absolute; top: 5px; left: 5px; background: rgba(0,0,0,0.7); padding: 2px 8px; font-size: 12px; border-radius: 4px; pointer-events: none;}
        .controls { background: #333; padding: 20px; border-radius: 8px; display: inline-block; margin-bottom: 20px; }
        .control-group { margin: 10px 0; display: flex; align-items: center; gap: 10px; }
        input[type=range] { width: 200px; }
        #status { color: #aaa; margin-bottom: 10px; font-style: italic; }
    </style>
</head>
<body>

    <h2>Mel Spectrogram Analysis</h2>
    <div id="status">Loading OpenCV.js... (this may take a moment)</div>

    <div class="controls">
        <div class="control-group">
            <label for="thresh1">Threshold 1:</label>
            <input type="range" id="thresh1" min="0" max="500" value="100">
            <span id="val1">100</span>
        </div>
        <div class="control-group">
            <label for="thresh2">Threshold 2:</label>
            <input type="range" id="thresh2" min="0" max="500" value="200">
            <span id="val2">200</span>
        </div>
    </div>

    <div class="container">
        <div class="canvas-wrapper">
            <div class="label">Mel Spectrogram</div>
            <img id="sourceImage" src="{{IMG_DATA}}" style="display:none;" onload="initCV()" />
            <canvas id="canvasInput"></canvas>
        </div>

        <div class="canvas-wrapper">
            <div class="label">Canny Edges</div>
            <canvas id="canvasOutput"></canvas>
        </div>
    </div>

    <script async src="https://docs.opencv.org/4.8.0/opencv.js" onload="onOpenCvReady();" type="text/javascript"></script>

    <script type="text/javascript">
        let cvReady = false;
        let imgLoaded = false;

        function onOpenCvReady() {
            cv['onRuntimeInitialized'] = () => {
                document.getElementById('status').innerText = "OpenCV Ready. Processing...";
                cvReady = true;
                processImage();
            };
        }

        function initCV() {
            imgLoaded = true;
            processImage();
        }

        function processImage() {
            if (!cvReady || !imgLoaded) return;

            const imgElement = document.getElementById('sourceImage');
            const canvasInput = document.getElementById('canvasInput');
            const ctxInput = canvasInput.getContext('2d');

            canvasInput.width = imgElement.width;
            canvasInput.height = imgElement.height;
            ctxInput.drawImage(imgElement, 0, 0);

            applyCanny();
        }

        function applyCanny() {
            if (!cvReady) return;

            const t1 = parseInt(document.getElementById('thresh1').value);
            const t2 = parseInt(document.getElementById('thresh2').value);
            
            document.getElementById('val1').innerText = t1;
            document.getElementById('val2').innerText = t2;

            let src = cv.imread('canvasInput');
            let dst = new cv.Mat();
            
            cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
            cv.Canny(src, dst, t1, t2, 3, false);
            cv.imshow('canvasOutput', dst);
            
            src.delete();
            dst.delete();
        }

        document.getElementById('thresh1').addEventListener('input', applyCanny);
        document.getElementById('thresh2').addEventListener('input', applyCanny);
    </script>
</body>
</html>
    """

    # 6. 替换占位符并写入文件
    final_html = html_template.replace("{{IMG_DATA}}", img_src)
    
    # 自动根据输入文件名生成输出文件名
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{base_name}_canny.html"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Success! Open '{output_filename}' in your browser.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_wav_file>")
    else:
        create_mel_html(sys.argv[1])
