"""Local demo web page: input -> real model -> output.

Run:  python scripts/demo_web.py
Then open http://localhost:8500

Uses the real trained pipeline from models/model.joblib. No mock data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.config import MODELS_DIR
from src.predict import InvalidAppointmentError, load_model, predict_one

app = FastAPI()
META = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))


class Req(BaseModel):
    scheduled_day: str
    appointment_day: str
    age: int
    gender: str
    neighbourhood: str
    scholarship: int = 0
    sms_received: int = 0
    prior_appointments: int = 0
    prior_noshows: int = 0


@app.post("/api/predict")
def api_predict(r: Req):
    record = r.model_dump()
    prior = record["prior_appointments"]
    record["prior_noshow_rate"] = record["prior_noshows"] / prior if prior else -1.0
    record["is_first_appointment"] = int(prior == 0)
    record["days_since_last_appointment"] = -1.0
    try:
        out = predict_one(record)
    except InvalidAppointmentError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    out["input_echo"] = record
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    m = META["test_metrics"]
    return (
        PAGE.replace("__F1__", f"{m['f1']:.3f}")
        .replace("__REC__", f"{m['recall']:.1%}")
        .replace("__PREC__", f"{m['precision']:.1%}")
        .replace("__ROC__", f"{m['roc_auc']:.3f}")
        .replace("__THR__", f"{META['threshold']:.3f}")
    )


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>No-Show Prediction - Live Demo</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,Segoe UI,sans-serif;background:#0f172a;color:#e2e8f0}
.wrap{max-width:1150px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px}
.sub{color:#94a3b8;font-size:13px;margin-bottom:18px}
.metrics{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px}
.metric{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:8px 14px}
.metric b{display:block;font-size:19px;color:#38bdf8}
.metric span{font-size:11px;color:#94a3b8;text-transform:uppercase}
.flow{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px}
.card h2{font-size:12px;text-transform:uppercase;color:#94a3b8;margin:0 0 12px;letter-spacing:.5px}
label{display:block;font-size:11px;color:#94a3b8;margin:8px 0 3px}
input,select{width:100%;padding:7px 9px;border-radius:6px;border:1px solid #475569;
background:#0f172a;color:#e2e8f0;font-size:13px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{width:100%;margin-top:14px;padding:11px;border:0;border-radius:8px;
background:#0284c7;color:#fff;font-size:15px;font-weight:600;cursor:pointer}
button:hover{background:#0369a1}
.presets{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.presets button{width:auto;margin:0;padding:5px 10px;font-size:11px;background:#334155}
.presets button:hover{background:#475569}
.step{padding:8px 10px;margin-bottom:6px;background:#0f172a;border-left:3px solid #334155;
border-radius:4px;font-size:12px;color:#64748b;transition:.25s}
.step.on{border-left-color:#38bdf8;color:#e2e8f0}
.step b{color:#38bdf8}
.gauge{height:26px;background:#0f172a;border-radius:13px;overflow:hidden;margin:10px 0;
border:1px solid #334155}
.fill{height:100%;width:0;transition:width .6s;display:flex;align-items:center;
justify-content:flex-end;padding-right:9px;font-weight:700;font-size:13px;color:#fff}
.tier{display:inline-block;padding:5px 14px;border-radius:20px;font-weight:700;font-size:14px}
.big{font-size:40px;font-weight:800;text-align:center;margin:6px 0}
.rec{background:#0f172a;padding:10px;border-radius:6px;font-size:13px;margin-top:10px;
border:1px solid #334155}
pre{background:#0f172a;padding:10px;border-radius:6px;font-size:11px;overflow-x:auto;
color:#94a3b8;max-height:190px;border:1px solid #334155}
.err{background:#7f1d1d;color:#fecaca;padding:10px;border-radius:6px;font-size:12px}
@media(max-width:900px){.flow{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<h1>No-Show Prediction &mdash; Live Demo</h1>
<div class="sub">XGBoost + isotonic calibration &middot; real model loaded from models/model.joblib</div>

<div class="metrics">
<div class="metric"><b>__F1__</b><span>Test F1</span></div>
<div class="metric"><b>__REC__</b><span>Recall</span></div>
<div class="metric"><b>__PREC__</b><span>Precision</span></div>
<div class="metric"><b>__ROC__</b><span>ROC-AUC</span></div>
<div class="metric"><b>__THR__</b><span>Threshold</span></div>
</div>

<div class="flow">
  <div class="card">
    <h2>1 &middot; Input</h2>
    <div class="presets">
      <button onclick="preset('high')">High risk</button>
      <button onclick="preset('low')">Low risk</button>
      <button onclick="preset('bad')">Invalid</button>
    </div>
    <div class="row">
      <div><label>Booked on</label><input id="sched" type="date" value="2016-05-02"></div>
      <div><label>Appointment</label><input id="appt" type="date" value="2016-05-30"></div>
    </div>
    <div class="row">
      <div><label>Age</label><input id="age" type="number" value="22"></div>
      <div><label>Gender</label><select id="gender"><option>F</option><option>M</option></select></div>
    </div>
    <label>Neighbourhood</label><input id="hood" value="JARDIM CAMBURI">
    <div class="row">
      <div><label>Past visits</label><input id="prior" type="number" value="0"></div>
      <div><label>Past missed</label><input id="missed" type="number" value="0"></div>
    </div>
    <div class="row">
      <div><label>Welfare</label><select id="scho"><option value="0">No</option><option value="1">Yes</option></select></div>
      <div><label>SMS sent</label><select id="sms"><option value="0">No</option><option value="1">Yes</option></select></div>
    </div>
    <button onclick="run()">Predict &rarr;</button>
  </div>

  <div class="card">
    <h2>2 &middot; Pipeline</h2>
    <div class="step" id="s1"><b>Validate</b><br>required fields, ranges, date order</div>
    <div class="step" id="s2"><b>Feature engineering</b><br>lead time, weekday, age group, history</div>
    <div class="step" id="s3"><b>Preprocess</b><br>StandardScaler + OneHotEncoder</div>
    <div class="step" id="s4"><b>XGBoost</b><br>400 trees, depth 5</div>
    <div class="step" id="s5"><b>Calibrate</b><br>isotonic &rarr; true probability</div>
    <div class="step" id="s6"><b>Risk tier</b><br>validation quantiles</div>
    <h2 style="margin-top:14px">Features sent to model</h2>
    <pre id="echo">-</pre>
  </div>

  <div class="card">
    <h2>3 &middot; Output</h2>
    <div id="out" style="color:#64748b;font-size:13px">Press Predict.</div>
  </div>
</div>
</div><script>
function preset(k){
  var v={
    high:{sched:'2016-05-02',appt:'2016-05-30',age:22,g:'F',h:'JARDIM CAMBURI',p:5,m:5,s:1,sms:1},
    low :{sched:'2016-05-30',appt:'2016-05-30',age:65,g:'M',h:'CENTRO',p:5,m:0,s:0,sms:0},
    bad :{sched:'2016-06-30',appt:'2016-05-30',age:999,g:'F',h:'CENTRO',p:0,m:0,s:0,sms:0}
  }[k];
  sched.value=v.sched; appt.value=v.appt; age.value=v.age; gender.value=v.g;
  hood.value=v.h; prior.value=v.p; missed.value=v.m; scho.value=v.s; sms.value=v.sms;
}
async function run(){
  var ids=['s1','s2','s3','s4','s5','s6'];
  ids.forEach(function(i){document.getElementById(i).classList.remove('on')});
  var out=document.getElementById('out');
  out.innerHTML='<div style="color:#64748b">Running...</div>';
  var body={scheduled_day:sched.value,appointment_day:appt.value,
    age:+age.value,gender:gender.value,neighbourhood:hood.value,
    scholarship:+scho.value,sms_received:+sms.value,
    prior_appointments:+prior.value,prior_noshows:+missed.value};
  for(var i=1;i<=6;i++){
    await new Promise(function(r){setTimeout(r,110)});
    document.getElementById('s'+i).classList.add('on');
  }
  var res=await fetch('/api/predict',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  var d=await res.json();
  if(!res.ok){
    ['s2','s3','s4','s5','s6'].forEach(function(i){
      document.getElementById(i).classList.remove('on')});
    var msg=d.error||(d.detail&&d.detail[0]&&d.detail[0].msg)||'invalid input';
    out.innerHTML='<div class="err"><b>Rejected at validation</b><br>'+msg+'</div>'+
      '<div class="rec">No prediction is returned for invalid input - '+
      'the system refuses rather than guessing.</div>';
    document.getElementById('echo').textContent='-';
    return;
  }
  var c={High:'#dc2626',Medium:'#f59e0b',Low:'#16a34a'}[d.risk_tier];
  out.innerHTML=
    '<div class="big" style="color:'+c+'">'+d.no_show_percentage+'%</div>'+
    '<div style="text-align:center"><span class="tier" style="background:'+c+'">'+
      d.risk_tier+' RISK</span></div>'+
    '<div class="gauge"><div class="fill" id="f" style="background:'+c+'"></div></div>'+
    '<div class="rec"><b>Action:</b><br>'+d.recommendation+'</div>'+
    '<div class="rec"><b>Lead time:</b> '+d.lead_time_days+' days<br>'+
      '<b>Threshold:</b> '+d.threshold_used.toFixed(3)+'<br>'+
      '<b>Model:</b> '+d.model+' (calibrated)</div>';
  setTimeout(function(){
    var f=document.getElementById('f');
    f.style.width=Math.max(d.no_show_percentage,7)+'%';
    f.textContent=d.no_show_percentage+'%';},60);
  document.getElementById('echo').textContent=JSON.stringify(d.input_echo,null,1);
}
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn

    load_model()
    print("\n  Open in browser:  http://localhost:8500\n")
    uvicorn.run(app, host="127.0.0.1", port=8500, log_level="warning")
