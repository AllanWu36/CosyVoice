import os
import sys
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed
import torchaudio
from concurrent.futures import ThreadPoolExecutor
import uuid
import threading
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

sft_spk = []

cosyvoice = None
class SynthesizeRequest(BaseModel):
    text: str
    spk_id: str
    stream: Optional[bool] = False
    speed: Optional[float] = 1.0
    seed: Optional[int] = 0

class SynthesizeResponse(BaseModel):
    wav_path: List[str]
    sample_rate: int

executor = ThreadPoolExecutor(max_workers=5)
task_store = {}  # {task_id: {"status": ..., "result": ...}}
task_store_lock = threading.Lock()  # 保证多线程安全

def run_synthesize_task(task_id, req_dict):
    try:
        with task_store_lock:
            task_store[task_id]["status"] = "running"
        # 参数还原为对象
        text = req_dict["text"]
        spk_id = req_dict["spk_id"]
        stream = req_dict.get("stream", False)
        speed = req_dict.get("speed", 1.0)
        seed = req_dict.get("seed", 0)
        if spk_id not in sft_spk:
            raise ValueError(f"音色 {spk_id} 不在可用列表中")
        set_all_random_seed(seed)
        output_dir = 'runtime/audio_outputs'
        os.makedirs(output_dir, exist_ok=True)
        file_list = []
        prompt_speech_16k = load_wav('./test.wav', 16000)
        prompt_text = '百度百科是百度公司在线运营的一部半开放式中文网络百科全书。'
        # assert cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, 'my_zero_shot_spk') is True
        # for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
        #     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
        # cosyvoice.save_spkinfo()
        #完成存储
        # sft_spk = cosyvoice.list_available_spks()
        # for spk in sft_spk:
        #     logging.info(spk)
        # if spk_id not in sft_spk:
        #     raise ValueError(f"音色 {spk_id} 不在可用列表中")
        # spk_id = 'my_zero_shot_spk'
        for i, result in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech_16k, stream=False)):
        # for i, result in enumerate(cosyvoice.inference_sft(text, spk_id, stream=False, speed=speed)):
            audio = result['tts_speech']
            if audio is None:
                raise RuntimeError('语音合成失败')
            wav_path = os.path.join(output_dir, f"tts_{spk_id}_{abs(hash(text)) % 100000}_{i}.wav")
            torchaudio.save(wav_path, audio, cosyvoice.sample_rate)
            file_list.append(wav_path)
        with task_store_lock:
            task_store[task_id]["status"] = "completed"
            task_store[task_id]["result"] = {
                "wav_path": file_list,
                "sample_rate": cosyvoice.sample_rate
            }
    except Exception as e:
        #打印调用栈
        import traceback
        traceback.print_exc()
        with task_store_lock:
            task_store[task_id]["status"] = "failed"
            task_store[task_id]["result"] = str(e)

@app.get('/list_spks', response_model=List[str])
def list_spks():
    """获取所有可用的预设音色名"""
    return sft_spk

@app.post('/synthesize')
def synthesize(req: SynthesizeRequest):
    task_id = str(uuid.uuid4())
    with task_store_lock:
        task_store[task_id] = {"status": "pending", "result": None}
    # 转为dict，避免pydantic对象线程不安全
    executor.submit(run_synthesize_task, task_id, req.dict())
    return {"task_id": task_id}

@app.get('/synthesize_status/{task_id}')
def synthesize_status(task_id: str):
    with task_store_lock:
        task = task_store.get(task_id)
        if not task:
            return {"error": "task not found"}
        return {"status": task["status"], "result": task["result"]}

if __name__ == '__main__':
    # 初始化模型（可根据需要切换 CosyVoice 或 CosyVoice2）
    MODEL_DIR = '/home/wyh/git/CosyVoice/pretrained_models/CosyVoice2-0.5B'
    # MODEL_DIR = '/root/git/CosyVoice/pretrained_models/CosyVoice2-0.5B'

    try:
        cosyvoice = CosyVoice(MODEL_DIR)
    except Exception:
        try:
            cosyvoice = CosyVoice2(MODEL_DIR,load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
        except Exception:
            raise TypeError('no valid model_type!')

    # 获取可用音色
    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    uvicorn.run(app, host="0.0.0.0", port=50000)