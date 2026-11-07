"""Training-derived Studio presets; never exposes official test frames."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .data import load_pose

def presets(run: Path, scene: Path) -> list[dict]:
    protocol=json.loads((run/'frozen_protocol.json').read_text(encoding='utf-8'))
    if protocol.get('test_frames_not_used_for_selection'):
        raise ValueError('test poses are forbidden in Studio presets')
    output=[]
    for kind,key in (('training','train_frames'),('validation','validation_frames')):
        for name in protocol.get(key,[]):
            frame=Path(name); pose=load_pose(scene/frame.parent/frame.name.replace('.color.png','.pose.txt'))
            if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]): raise ValueError('invalid preset pose')
            output.append({'id':f'{kind}:{frame.as_posix()}','kind':kind,'frame':frame.as_posix(),'sequence':frame.parts[0],'pose':pose.tolist()})
    return output
