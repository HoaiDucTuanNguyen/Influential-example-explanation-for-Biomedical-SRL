import os, math, json, random, argparse, itertools, collections, time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification

def q0(a=13):
    random.seed(a); np.random.seed(a); torch.manual_seed(a)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a)

def q1(x):
    return x.detach().float().cpu()

def q2(z):
    r=[]
    i=0
    while i<len(z):
        v=z[i]
        if v.startswith("B-"):
            t=v[2:]; j=i+1
            while j<len(z) and z[j]==("I-"+t): j+=1
            r.append((i,j,t)); i=j
        elif v.startswith("I-"):
            t=v[2:]; j=i+1
            while j<len(z) and z[j]==("I-"+t): j+=1
            r.append((i,j,t)); i=j
        else:
            i+=1
    return r

def q3(a,b):
    return {i:j for i,j in enumerate(b)}

def q4(x):
    y={}
    for i,w in enumerate(x):
        if w is not None: y.setdefault(w,[]).append(i)
    return y

def q5(x, a, b, n=5):
    l=max(0,a-n); r=min(len(x),b+n)
    return x[l:r]

def q6(x):
    return " ".join(x).lower()

def q7(a,b):
    x=set(a); y=set(b)
    if not x and not y: return 0.0
    return len(x&y)/(len(x|y)+1e-9)

def q8(a,b):
    return (2.0*len(set(a)&set(b))+0.0)/(len(set(a))+len(set(b))+1e-9)

def q9(a,b):
    return 0.65*q7(a,b)+0.35*q8(a,b)

def qa(x):
    m=x.mean(0,keepdim=True)
    s=x.std(0,keepdim=True).clamp_min(1e-5)
    return (x-m)/s

def qb(a,b):
    a=qa(a); b=qa(b)
    c=b.transpose(0,1).matmul(a)
    try:
        u,_,vh=torch.linalg.svd(c,full_matrices=False)
        r=u.matmul(vh)
        return a-b.matmul(r)
    except Exception:
        return a-b

def qc(w,t):
    v=[]
    for z in (("B-"+t),("I-"+t)):
        if z in w: v.append(w[z])
    if not v: return None
    return torch.stack(v,0).mean(0)

def qd(u,w):
    a=[]
    for z in ["PER","ORG","LOC","MISC"]:
        x=qc(w,z)
        if x is not None: a.append(x/(x.norm()+1e-8))
    return torch.stack(a,0).mean(0) if a else torch.ones(u)/math.sqrt(u)

def qe(h, s, wt):
    a,b,t=s
    if b<=a: return None
    bb=h[a]; ee=h[b-1]
    if b-a<=2:
        cc=torch.zeros_like(bb)
    else:
        inn=h[a+1:b-1]
        if wt is None:
            ww=torch.ones(inn.shape[0],device=inn.device)/max(1,inn.shape[0])
        else:
            sc=F.relu(inn.matmul(wt)/(wt.norm()+1e-8))
            ww=sc/(sc.sum()+1e-8)
        cc=(ww[:,None]*inn).sum(0)
    return torch.cat([bb,ee,cc],0)

def qf(z, idx, sp, typ, dev):
    e=z["enc"].to(dev)
    y=z["wid"]
    m=q4(y)
    a,b,_=sp
    if a not in m or b-1 not in m: return None
    r=[]
    for w in range(a,b):
        if w in m: r+=m[w]
    if not r: return None
    return e, r, m.get(a,[]), m.get(b-1,[])

def qg(logits, ids, typ, lm, dev):
    v=[]
    if typ=="O":
        if "O" in lm: return logits[ids,lm["O"]].mean()
        return logits[ids].logsumexp(-1).mean()*0.0
    b=lm.get("B-"+typ,None); i=lm.get("I-"+typ,None)
    if b is None and i is None: return logits[ids].mean()*0.0
    if len(ids)==1:
        if b is not None: return logits[ids[0],b]
        return logits[ids[0],i]
    if b is not None: v.append(logits[ids[0],b])
    if i is not None: v.append(logits[ids[1:],i].mean())
    return torch.stack(v).mean()

def qh(logits, ids, lm):
    keys=["O","PER","ORG","LOC","MISC"]
    vals=torch.stack([qg(logits,ids,k,lm,logits.device) for k in keys],0)
    return keys, torch.softmax(vals,0)

def qi(keys, pr, gold):
    gi=keys.index(gold) if gold in keys else keys.index("O")
    j=torch.argmax(torch.cat([pr[:gi],pr[gi+1:]])).item()
    jj=j if j<gi else j+1
    return pr[gi]-pr[jj]

class qj:
    def __init__(self, ft, pt, tk, lab, dev):
        self.ft=ft; self.pt=pt; self.tk=tk; self.lab=lab; self.dev=dev
        self.lm={v:k for k,v in lab.items()} if isinstance(lab,dict) else {v:i for i,v in enumerate(lab)}
        self.im={i:v for v,i in self.lm.items()}
        self.hw=ft.classifier.weight.detach().to(dev)
        self.tw={t:qc(self.hw,self.lm,t) for t in ["PER","ORG","LOC","MISC"]}
        self.dw=qd(self.hw.shape[1],self.lm).to(dev)
        self.L=getattr(ft.config,"num_hidden_layers",12)
    def enc(self, words):
        x=self.tk(words,is_split_into_words=True,return_tensors="pt",truncation=True,max_length=192)
        y=x.word_ids(0)
        return {"enc":{k:v[0] for k,v in x.items()},"wid":y}
    @torch.no_grad()
    def run(self, z):
        x={k:v.unsqueeze(0).to(self.dev) for k,v in z["enc"].items()}
        a=self.ft(**x,output_hidden_states=True,return_dict=True)
        b=self.pt(**x,output_hidden_states=True,return_dict=True)
        return a,b
    @torch.no_grad()
    def atv(self,z,sp,typ):
        a,b=self.run(z)
        y=z["wid"]; m=q4(y)
        r=[]
        for L in range(1,self.L+1):
            hf=a.hidden_states[L][0]
            hp=b.hidden_states[L][0]
            g=[]
            p=[]
            ks=sorted(m.keys())
            for w in ks:
                g.append(hf[m[w]].mean(0)); p.append(hp[m[w]].mean(0))
            g=torch.stack(g,0); p=torch.stack(p,0)
            d=qb(g,p)
            mm={w:i for i,w in enumerate(ks)}
            if sp[0] not in mm or sp[1]-1 not in mm: r.append(None); continue
            h=[]
            for w in range(sp[0],sp[1]):
                if w in mm: h.append(d[mm[w]])
            h=torch.stack(h,0)
            wt=self.tw.get(typ,None)
            if wt is None: wt=self.dw
            r.append(qe(h,(0,h.shape[0],typ),wt))
        return r
    @torch.no_grad()
    def logits(self,z):
        x={k:v.unsqueeze(0).to(self.dev) for k,v in z["enc"].items()}
        return self.ft(**x,return_dict=True).logits[0]
    def fac(self,z,sp,tvec,layer,dist,lmbd,gold):
        e,ids,bd,ed=qf(z,0,sp,gold,self.dev)
        if e is None: return None
        b0=self.logits(z)
        k0,p0=qh(b0,ids,self.lm)
        m0=qi(k0,p0,gold)
        if tvec is None: return 0.0
        tv=tvec.to(self.dev)
        d=tv.shape[0]//3
        aa=tv[:d]; bb=tv[d:2*d]; cc=tv[2*d:]
        ww=1.0/(1.0+float(dist))
        y=z["wid"]; mp=q4(y)
        a,b,_=sp
        inner=[w for w in range(a+1,b-1) if w in mp]
        if inner:
            base=torch.stack([b0[mp[w]].mean(0) for w in inner],0)
            wv=self.tw.get(gold,None)
            if wv is None: v=torch.ones(len(inner),device=self.dev)/len(inner)
            else:
                hbase=torch.stack([self.run(z)[0].hidden_states[layer][0][mp[w]].mean(0) for w in inner],0)
                ss=F.relu(hbase.matmul(wv.to(self.dev))/(wv.norm().to(self.dev)+1e-8))
                v=ss/(ss.sum()+1e-8)
        else:
            v=None
        def hk(mod,inp,out):
            h=out[0].clone()
            for i in bd: h[0,i,:]-=lmbd*ww*aa
            for i in ed: h[0,i,:]-=lmbd*ww*bb
            if v is not None:
                for c,w in enumerate(inner):
                    for i in mp[w]: h[0,i,:]-=lmbd*ww*v[c]*cc
            return (h,)+out[1:]
        hdl=self.ft.bert.encoder.layer[layer-1].register_forward_hook(hk)
        try:
            b1=self.logits(z)
        finally:
            hdl.remove()
        k1,p1=qh(b1,ids,self.lm)
        m1=qi(k1,p1,gold)
        return float((m0-m1).detach().cpu())
    def pred(self,z,sp):
        e,ids,_,_=qf(z,0,sp,"O",self.dev)
        if e is None: return "O"
        k,p=qh(self.logits(z),ids,self.lm)
        return k[int(torch.argmax(p).item())]

def qk(ds,names,lim=None):
    out=[]
    c=0
    for ex in ds:
        tags=[names[i] for i in ex["ner_tags"]]
        spans=q2(tags)
        if spans:
            out.append((ex["tokens"],spans))
            c+=1
        if lim and c>=lim: break
    return out

def ql(Q,data,cap=None):
    r=[]
    for si,(w,ss) in enumerate(data):
        z=Q.enc(w)
        for sp in ss:
            ty=sp[2]
            a=Q.atv(z,sp,ty)
            if any(x is None for x in a): continue
            r.append({"i":si,"w":w,"z":z,"s":sp,"y":ty,"a":a,"ctx":q5(w,sp[0],sp[1])})
            if cap and len(r)>=cap: return r
    return r

def qm(Q,data,cap=None):
    r=[]
    for si,(w,ss) in enumerate(data):
        z=Q.enc(w)
        for sp in ss:
            ty=sp[2]
            pr=Q.pred(z,sp)
            a=Q.atv(z,sp,ty)
            if any(x is None for x in a): continue
            r.append({"i":si,"w":w,"z":z,"s":sp,"y":ty,"p":pr,"ok":1 if pr==ty else -1,"a":a,"ctx":q5(w,sp[0],sp[1])})
            if cap and len(r)>=cap: return r
    return r

def qn(A,layer):
    x=torch.stack([v["a"][layer-1] for v in A],0)
    s=x.var(0,unbiased=False).clamp_min(1e-4)
    return x,s

def qo(x,Y,s):
    return ((Y-x[None,:])**2/s[None,:]).sum(1).sqrt()

def qp(T,R,ty,forbid,k,m,seed):
    C=[i for i,r in enumerate(R) if r["y"]==ty and i not in forbid]
    if len(C)<=m: return C
    z=q6(T)
    vals=[]
    for i in C:
        vals.append((q9(z.split(),q6(R[i]["ctx"]).split()),i))
    vals.sort(reverse=True)
    pool=[i for _,i in vals[:min(len(vals),max(m*4,m+k+25))]]
    rng=random.Random(seed)
    return rng.sample(pool,m) if len(pool)>m else pool

def qr(Q,TR,TE,args,layer):
    X,S=qn(TR,layer)
    R=[]
    for n,t in enumerate(TE):
        x=t["a"][layer-1]
        D=qo(x,X,S)
        same=[i for i,u in enumerate(TR) if u["y"]==t["y"]]
        if not same: continue
        ds=torch.tensor([D[i] for i in same])
        oo=torch.argsort(ds)[:args.k].tolist()
        rr=[same[i] for i in oo]
        forb=set(rr)
        ref=qp(t["ctx"],TR,t["y"],forb,args.k,args.m,args.seed+n)
        if len(ref)<1: continue
        pol_r=[]
        for idx in rr:
            pol_r.append(abs(Q.fac(t["z"],t["s"],TR[idx]["a"][layer-1],layer,float(D[idx]),args.lam,t["y"])))
        pol_b=[]
        for idx in ref:
            pol_b.append(abs(Q.fac(t["z"],t["s"],TR[idx]["a"][layer-1],layer,float(D[idx]),args.lam,t["y"])))
        if not pol_r or not pol_b: continue
        a=0.0
        for u in pol_r:
            a+=sum(1 for v in pol_b if v<u)/len(pol_b)
        a/=len(pol_r)
        sg=[]
        for idx in rr:
            z=Q.fac(t["z"],t["s"],TR[idx]["a"][layer-1],layer,float(D[idx]),args.lam,t["y"])
            sg.append(1 if (z>0 and t["ok"]>0) or (z<0 and t["ok"]<0) else 0)
        R.append((a,float(np.mean(sg)) if sg else 0.0))
    if not R: return {"afi":None,"ppa":None,"n":0}
    return {"afi":float(np.mean([x[0] for x in R])*100.0),"ppa":float(np.mean([x[1] for x in R])*100.0),"n":len(R)}

def qs(v,B=500,seed=17):
    rng=np.random.default_rng(seed)
    v=np.array(v,dtype=float)
    if len(v)<2: return [float(v.mean()),float(v.mean())]
    a=[]
    for _ in range(B): a.append(rng.choice(v,len(v),replace=True).mean())
    return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]

def qt():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ft",default="dslim/bert-base-NER")
    ap.add_argument("--pt",default="bert-base-cased")
    ap.add_argument("--k",type=int,default=25)
    ap.add_argument("--m",type=int,default=100)
    ap.add_argument("--lam",type=float,default=0.35)
    ap.add_argument("--seed",type=int,default=17)
    ap.add_argument("--train_sentences",type=int,default=1200)
    ap.add_argument("--test_sentences",type=int,default=250)
    ap.add_argument("--max_train_spans",type=int,default=4000)
    ap.add_argument("--max_test_spans",type=int,default=120)
    ap.add_argument("--layers",default="9,10")
    ap.add_argument("--out",default="conll2003_atavee_ner.json")
    return ap.parse_args()

def qu():
    a=qt(); q0(a.seed)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D=load_dataset("conll2003")
    names=D["train"].features["ner_tags"].feature.names
    tk=AutoTokenizer.from_pretrained(a.ft,use_fast=True)
    ft=AutoModelForTokenClassification.from_pretrained(a.ft).to(dev).eval()
    pt=AutoModel.from_pretrained(a.pt).to(dev).eval()
    Q=qj(ft,pt,tk,ft.config.id2label,dev)
    tr=qk(D["train"],names,a.train_sentences)
    te=qk(D["test"],names,a.test_sentences)
    TR=ql(Q,tr,a.max_train_spans)
    TE=qm(Q,te,a.max_test_spans)
    layers=[int(x) for x in a.layers.split(",") if x.strip()]
    o={"config":vars(a),"device":str(dev),"train_spans":len(TR),"test_spans":len(TE),"result":{}}
    for L in layers:
        o["result"][str(L)]=qr(Q,TR,TE,a,L)
    with open(a.out,"w",encoding="utf-8") as f: json.dump(o,f,ensure_ascii=False,indent=2)
    print(json.dumps(o,ensure_ascii=False,indent=2))

if __name__=="__main__":
    qu()
