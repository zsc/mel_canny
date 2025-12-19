import sys
import base64
import io
import numpy as np
import librosa
from PIL import Image

def process_audio_to_html(audio_path, output_html="mel_skeleton.html"):
    print(f"Loading audio: {audio_path}...")
    
    # 1. 音频处理与 Mel 谱图计算
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)

    # 计算 Mel Spectrogram
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # 转为对数刻度 (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 2. 图像预处理 (归一化到 0-255)
    # 翻转 Y 轴，因为 spectrogram 默认低频在下，但在图像矩阵中索引 0 在上
    S_dB = np.flipud(S_dB)
    
    img_min = S_dB.min()
    img_max = S_dB.max()
    norm_S = (S_dB - img_min) / (img_max - img_min) * 255.0
    img_data = norm_S.astype(np.uint8)

    # 3. 将图像转换为 Base64 字符串
    img = Image.fromarray(img_data, mode='L') # L mode is grayscale
    # 调整一下大小，避免太长导致浏览器处理太慢，限制最大宽度
    max_width = 1200
    if img.width > max_width:
        new_height = int(img.height * (max_width / img.width))
        img = img.resize((max_width, new_height), Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 4. 生成 HTML
    # 这里包含了完整的 CSS 和 JS (Zhang-Suen 算法实现)
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mel Spectrum Skeletonization</title>
    <style>
        body {{ font-family: sans-serif; background: #1e1e1e; color: #ddd; margin: 0; padding: 20px; }}
        h2 {{ margin-bottom: 10px; border-bottom: 1px solid #444; padding-bottom: 10px; }}
        .controls {{ 
            background: #2d2d2d; padding: 15px; border-radius: 8px; margin-bottom: 20px; 
            display: flex; gap: 30px; align-items: center; flex-wrap: wrap;
        }}
        .control-group {{ display: flex; flex-direction: column; gap: 5px; }}
        label {{ font-size: 0.9em; color: #aaa; }}
        input[type=range] {{ width: 200px; cursor: pointer; }}
        span.val {{ color: #00bcd4; font-weight: bold; width: 40px; display: inline-block; text-align: right; }}
        
        .container {{ display: flex; flex-direction: column; gap: 20px; }}
        .canvas-wrapper {{ position: relative; width: 100%; overflow-x: auto; }}
        canvas {{ display: block; background: #000; border: 1px solid #444; }}
        
        .row {{ display: flex; gap: 20px; }}
        .panel {{ flex: 1; }}
        .panel h3 {{ font-size: 1rem; color: #888; margin: 5px 0; }}
        
        /* 隐藏原始图片，仅用于加载数据 */
        #sourceImg {{ display: none; }}
        
        .status {{ color: #ff9800; font-size: 0.8em; margin-left: 10px; }}
    </style>
</head>
<body>

    <h2>Mel Spectrum Ridge Extraction (Skeletonization)</h2>

    <div class="controls">
        <div class="control-group">
            <label>Binary Threshold <span id="threshVal" class="val">100</span></label>
            <input type="range" id="threshSlider" min="0" max="255" value="80">
        </div>
        <div class="control-group">
            <label>Horizontal Bias (Pre-connect) <span id="biasVal" class="val">1</span></label>
            <input type="range" id="biasSlider" min="0" max="10" value="2">
        </div>
        <div class="control-group">
            <button onclick="runProcessing()" style="padding: 8px 16px; cursor: pointer; background: #00bcd4; border: none; border-radius: 4px; color: white; font-weight: bold;">Update / Run</button>
            <span id="status" class="status">Ready</span>
        </div>
    </div>

    <div class="row">
        <div class="panel">
            <h3>Original Mel Spectrogram</h3>
            <canvas id="canvasOrg"></canvas>
        </div>
        <div class="panel">
            <h3>Extracted Ridges (Thinning Algorithm)</h3>
            <canvas id="canvasRes"></canvas>
        </div>
    </div>

    <img id="sourceImg" src="data:image/png;base64,{img_base64}" onload="init()" />

<script>
    const img = document.getElementById('sourceImg');
    const cOrg = document.getElementById('canvasOrg');
    const ctxOrg = cOrg.getContext('2d');
    const cRes = document.getElementById('canvasRes');
    const ctxRes = cRes.getContext('2d');
    
    const sliderThresh = document.getElementById('threshSlider');
    const sliderBias = document.getElementById('biasSlider');
    const statusSpan = document.getElementById('status');

    // 更新显示的数值
    sliderThresh.oninput = () => document.getElementById('threshVal').innerText = sliderThresh.value;
    sliderBias.oninput = () => document.getElementById('biasVal').innerText = sliderBias.value;
    
    // 监听变化自动运行 (Debounce 防止卡顿)
    let timeout;
    [sliderThresh, sliderBias].forEach(el => {{
        el.addEventListener('change', () => {{
            statusSpan.innerText = "Processing...";
            setTimeout(runProcessing, 10);
        }});
    }});

    function init() {{
        // 设置 Canvas 大小
        cOrg.width = img.width;
        cOrg.height = img.height;
        cRes.width = img.width;
        cRes.height = img.height;

        // 绘制原始图
        ctxOrg.drawImage(img, 0, 0);
        
        // 首次运行
        runProcessing();
    }}

    function runProcessing() {{
        const width = cOrg.width;
        const height = cOrg.height;
        
        // 1. 获取原始像素
        const frame = ctxOrg.getImageData(0, 0, width, height);
        const data = frame.data; // RGBA array
        
        const threshold = parseInt(sliderThresh.value);
        const hBias = parseInt(sliderBias.value);

        // 创建二值化矩阵 (0 or 1)
        // 我们使用一个平坦的数组来模拟二维矩阵，binaryMap[y * width + x]
        let binaryMap = new Uint8Array(width * height);

        for (let i = 0; i < data.length; i += 4) {{
            // 灰度值 (我们是灰度图，R=G=B)
            let val = data[i]; 
            binaryMap[i/4] = (val > threshold) ? 1 : 0;
        }}

        // 2. 预处理：横向增强 (Horizontal Morphological Dilation/Closing)
        // 为了连接那些因为能量波动断开的横向线条
        if (hBias > 0) {{
            let enhancedMap = new Uint8Array(width * height);
            for (let y = 0; y < height; y++) {{
                for (let x = 0; x < width; x++) {{
                    if (binaryMap[y * width + x] === 1) {{
                        // 如果当前点是白点，向左右扩散 hBias 个像素
                        for (let k = -hBias; k <= hBias; k++) {{
                            let nx = x + k;
                            if (nx >= 0 && nx < width) {{
                                enhancedMap[y * width + nx] = 1;
                            }}
                        }}
                    }}
                }}
            }}
            binaryMap = enhancedMap;
        }}

        // 3. 核心算法：Zhang-Suen Thinning (Skeletonization)
        // 这是一个迭代算法，直到图像不再改变
        let pixelsRemoved = true;
        let iterCount = 0;
        const maxIters = 100; // 防止死循环

        while (pixelsRemoved && iterCount < maxIters) {{
            pixelsRemoved = false;
            iterCount++;
            
            // Step 1
            let markers = [];
            for (let y = 1; y < height - 1; y++) {{
                for (let x = 1; x < width - 1; x++) {{
                    let p2 = binaryMap[(y-1)*width + x];
                    let p3 = binaryMap[(y-1)*width + x+1];
                    let p4 = binaryMap[y*width + x+1];
                    let p5 = binaryMap[(y+1)*width + x+1];
                    let p6 = binaryMap[(y+1)*width + x];
                    let p7 = binaryMap[(y+1)*width + x-1];
                    let p8 = binaryMap[y*width + x-1];
                    let p9 = binaryMap[(y-1)*width + x-1];

                    let A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                             (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                             (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                             (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                    
                    let B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    
                    let m1 = p2 * p4 * p6;
                    let m2 = p4 * p6 * p8;

                    if (binaryMap[y*width+x] === 1 && A === 1 && (B >= 2 && B <= 6) && m1 === 0 && m2 === 0) {{
                        markers.push(y*width+x);
                    }}
                }}
            }}
            
            if (markers.length > 0) pixelsRemoved = true;
            for (let i = 0; i < markers.length; i++) binaryMap[markers[i]] = 0;

            // Step 2
            markers = [];
            for (let y = 1; y < height - 1; y++) {{
                for (let x = 1; x < width - 1; x++) {{
                    let p2 = binaryMap[(y-1)*width + x];
                    let p3 = binaryMap[(y-1)*width + x+1];
                    let p4 = binaryMap[y*width + x+1];
                    let p5 = binaryMap[(y+1)*width + x+1];
                    let p6 = binaryMap[(y+1)*width + x];
                    let p7 = binaryMap[(y+1)*width + x-1];
                    let p8 = binaryMap[y*width + x-1];
                    let p9 = binaryMap[(y-1)*width + x-1];

                    let A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                             (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                             (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                             (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                    
                    let B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                    
                    let m1 = p2 * p4 * p8;
                    let m2 = p2 * p6 * p8;

                    if (binaryMap[y*width+x] === 1 && A === 1 && (B >= 2 && B <= 6) && m1 === 0 && m2 === 0) {{
                        markers.push(y*width+x);
                    }}
                }}
            }}

            if (markers.length > 0) pixelsRemoved = true;
            for (let i = 0; i < markers.length; i++) binaryMap[markers[i]] = 0;
        }}

        // 4. 将结果绘制回 Canvas
        // 创建一个新的 ImageData 对象
        let outputImg = ctxRes.createImageData(width, height);
        let outData = outputImg.data;

        for (let i = 0; i < width * height; i++) {{
            let val = binaryMap[i] * 255; 
            // 绿色显示骨架
            outData[i*4 + 0] = 0;   // R
            outData[i*4 + 1] = val; // G
            outData[i*4 + 2] = 0;   // B
            outData[i*4 + 3] = 255; // Alpha
        }}
        
        ctxRes.putImageData(outputImg, 0, 0);
        statusSpan.innerText = "Done (Iterations: " + iterCount + ")";
    }}
</script>
</body>
</html>
    """

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Successfully generated: {output_html}")
    print("Please open this file in your browser.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wav2mel_thinning.py <input.wav>")
    else:
        process_audio_to_html(sys.argv[1])
