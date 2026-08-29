import { useState, useRef, useEffect, useCallback, useMemo } from "react";

const API="/api/proxy";
const MODEL="claude-sonnet-4-20250514";
const MAX_CASES=15;
const G_DARK="#2d5a27",G_MID="#4a7c3f",G_LIGHT="#e8f5e0",G_RES="#d4edcc",G_BOR="#5a8a4f",G_PALE="#f4faf2";
const BLACK="#1a1a1a",GRAY1="#333",GRAY2="#666",GRAY3="#999",GRAY4="#ccc",WHITE="#fff";

async function streamClaude(messages,system,onChunk){
  const res=await fetch(API,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:MODEL,max_tokens:1200,system,stream:true,messages})});
  if(!res.ok)throw new Error("API "+res.status);
  const reader=res.body.getReader();const dec=new TextDecoder();let buf="";
  while(true){const{done,value}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});const lines=buf.split("\n");buf=lines.pop();for(const line of lines){if(!line.startsWith("data: "))continue;const d=line.slice(6).trim();if(d==="[DONE]")return;try{const e=JSON.parse(d);if(e.type==="content_block_delta"&&e.delta?.type==="text_delta")onChunk(e.delta.text);}catch{}}}
}
async function callJSON(prompt,system){
  const res=await fetch(API,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:MODEL,max_tokens:1500,system,messages:[{role:"user",content:prompt}]})});
  if(!res.ok)throw new Error("API "+res.status);
  const d=await res.json();return d.content[0].text;
}
async function scoreInput(text,mod){
  try{const raw=await callJSON(text,"Score this "+mod+" consulting input 0-100. Return ONLY JSON: {\"score\":0-100,\"gaps\":[\"gap1\"],\"strengths\":[\"s1\"],\"calibration\":\"directive|guided|collaborative|challenging\"}");return JSON.parse(raw.match(/\{[\s\S]*\}/)?.[0]);}
  catch{return{score:50,gaps:[],strengths:[],calibration:"guided"};}
}

function chunkText(text,size=480){const sents=text.match(/[^.!?\n]+[.!?\n]*/g)||[text];const chunks=[];let cur="";for(const s of sents){if((cur+s).length>size&&cur){chunks.push(cur.trim());cur=s;}else cur+=s;}if(cur.trim())chunks.push(cur.trim());return chunks.filter(c=>c.length>40);}
function buildVocab(chunks){const freq={};for(const c of chunks)for(const w of(c.toLowerCase().match(/\b\w{3,}\b/g)||[]))freq[w]=(freq[w]||0)+1;return Object.entries(freq).filter(([,f])=>f>1&&f<chunks.length*0.7).sort((a,b)=>b[1]-a[1]).slice(0,600).map(([w])=>w);}
function tfidf(text,vocab){const words=text.toLowerCase().match(/\b\w{3,}\b/g)||[];const freq={};for(const w of words)freq[w]=(freq[w]||0)+1;return vocab.map(v=>(freq[v]||0)/(words.length+1));}
function cosine(a,b){let dot=0,na=0,nb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]**2;nb+=b[i]**2;}return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);}
class KB{
  constructor(){this.docs=[];this.vocab=[];this.vecs=[];this.files=[];}
  add(name,text){const chunks=chunkText(text);this.docs.push(...chunks.map(c=>({text:c,source:name})));this.vocab=buildVocab(this.docs.map(d=>d.text));this.vecs=this.docs.map(d=>tfidf(d.text,this.vocab));if(!this.files.includes(name))this.files.push(name);}
  retrieve(q,k=4){if(!this.docs.length)return[];const qv=tfidf(q,this.vocab);return this.docs.map((d,i)=>({...d,score:cosine(qv,this.vecs[i])})).sort((a,b)=>b.score-a.score).slice(0,k).filter(d=>d.score>0.01);}
  get count(){return this.docs.length;}
}

const FW_SYSTEM=`You are a top McKinsey consultant. Return ONLY valid JSON for a MECE framework:
{"title":"PROBLEM LABEL MAX 6 WORDS","frameworkName":"Issue Tree | Porter's Five Forces | BCG Matrix | Value Chain | 4Ps | 7S | SWOT | Profitability Tree","rootLabel":"ROOT QUESTION MAX 5 WORDS","branches":[{"id":"b1","label":"BRANCH MAX 4 WORDS","color":"#2d5a27","children":[{"id":"b1c1","label":"SUB MAX 4 WORDS","color":"#4a7c3f","children":[{"id":"b1c1l1","label":"LEAF MAX 4 WORDS","color":"#4a7c3f","children":[]}]}]}],"meceCheck":"One sentence.","nonObviousAngle":"One unconventional insight.","keyHypothesis":"Single most important hypothesis."}
ALL labels UPPERCASE max 4 words. 2-4 branches. Max 3 levels. Return ONLY JSON.`;

const TREE_SYSTEM=`You are CO Consult. Return ONLY valid JSON for a market sizing tree:
{"problem":"PROBLEM IN UPPERCASE","assumptions":["A1","A2","A3","A4"],"tree":{"root":{"label":"POPULATION OF INDIA","value":"1.4 BILLION"},"level1":[{"label":"URBAN","value":"35%","sublabel":"490 MILLION"},{"label":"RURAL","value":"65%","sublabel":"910 MILLION"}],"level2":[{"parentIndex":0,"nodes":[{"label":"HIGH INCOME","value":"10%","sublabel":"49 MILLION"},{"label":"MIDDLE INCOME","value":"40%","sublabel":"196 MILLION"},{"label":"LOW INCOME","value":"50%","sublabel":"245 MILLION"}]}],"filters":[{"nodeLabel":"HIGH INCOME","chain":[{"label":"INTERNET PENETRATION","value":"80%","result":"39.2 MILLION"}]}],"convergence":{"label":"TOTAL TARGET USERS","value":"23.5 MILLION"},"calculation":[{"label":"TARGET USERS","value":"23.5 MILLION"},{"operator":"X","label":"PRICE","value":"RS. 500"},{"operator":"=","label":"MARKET SIZE","value":"RS. 141 BILLION","highlight":true}]},"formulaBar":[{"label":"POPULATION","operator":"X"},{"label":"PENETRATION","operator":"="},{"label":"RESULT","isResult":true}],"answer":"The estimated market size is RS. 141 BILLION.","bearCase":{"value":"RS. 90 BILLION"},"bullCase":{"value":"RS. 210 BILLION"},"sensitivityDriver":"Price assumption drives largest swing","difficultyDots":3}
India: Pop 1.4B, Mumbai 21.67M, Delhi 33.8M, Urban 35%. ALL labels UPPERCASE. ONLY JSON.`;

const SEED=`PROFITABILITY: Revenue minus Costs. Revenue equals Volume times Price.
MARKET ENTRY: Attractiveness, Competitive advantage, Entry mode, GTM, Financial case.
MECE: Mutually Exclusive Collectively Exhaustive. No overlaps. No gaps.
INDIA: Pop 1.4B, Urban 35%, GDP Rs 300L Cr, Delhi 33.8M, Mumbai 21.67M.
BCG: Stars invest, Cash Cows harvest, Question Marks selective, Dogs exit.
PORTER 5: Rivalry, Entrants, Substitutes, Buyer power, Supplier power.
UNIT ECONOMICS: LTV/CAC above 3x is healthy. CAC equals Marketing divided by Customers.
WACC: Weighted Average Cost of Capital. Use for DCF discount rate.
7S: Strategy Structure Systems Shared Values Skills Style Staff.
LBO: Leveraged Buyout. Returns via EBITDA growth deleveraging multiple expansion.
NPV: Positive NPV means value creating. IRR compare to hurdle rate.
INCOME: High 10%, Upper Middle 20%, Middle 40%, Low 30%.`;

const MODS=[
  {id:"case",icon:"◈",label:"Case Scoping"},
  {id:"framework",icon:"⊞",label:"Framework Engine"},
  {id:"sizing",icon:"∑",label:"Market Sizing"},
  {id:"financial",icon:"⌥",label:"Financial Analyst"},
  {id:"deck",icon:"▤",label:"Deck Builder"},
];

const QUICK={
  case:["Client revenue fell 20% — scope the problem","D2C brand wants to grow 3x in 2 years","PE firm evaluating edtech acquisition"],
  framework:["Why is customer churn high?","Should we enter the rural lending market?","How do we reduce costs for a retailer?"],
  sizing:["Market size of healthy biscuit industry in India","Number of Swiggy orders in Delhi in a day","Number of Spotify premium subscribers in India"],
  financial:["Unit economics for a SaaS at Rs 500 ARR","Break-even for a cloud kitchen in Mumbai","LTV/CAC for a D2C skincare brand"],
  deck:["Build the CEO report from my session work","Synthesise all findings into a presentation","Series A investor narrative for D2C brand"],
};

function buildSystem(mod,cal,score,rag,caseMem,sessions){

  const calGuide={
    directive:"SUGGEST MODE: Do NOT give full answers. Ask 1-2 sharp questions. Surface 2-3 options. Let the user lead.",
    guided:"GUIDE MODE: Walk step by step. ONE question at a time. Acknowledge what is right. Redirect gently.",
    collaborative:"COLLABORATIVE MODE: Fill key gaps. Challenge one assumption. Co-build the answer.",
    challenging:"SOLVE MODE: Give the COMPLETE answer immediately. Full analysis, all numbers, final recommendation. No clarifying questions.",
  };
  const prompts={
    case:`Mode: Case Scoping. A MECE framework diagram has been generated above.

If this is the user's FIRST message and it is brief or ambiguous, respond conversationally first:
- Acknowledge what you heard in 1-2 sentences
- Ask 2-3 clarifying questions like: Who is the client and what industry? What triggered this problem — is it sudden or gradual? What does success look like in 6 months?
- Keep it warm and natural. Do NOT produce the full structured output yet.

If the input is already detailed OR it is a follow-up message, write the full narrative below the diagram:

**Problem Definition**
- Client and industry
- Core question: what exactly are we solving
- Scope: what is in and out
- Success metric: how we measure it

**Stakeholder Map**
- Who cares about this problem and why

**Key Hypotheses**
- The 2-3 most likely root causes, ranked by probability

**What I Would Do Next**
- Specific next module and why
- One thing most people miss in cases like this

⚡ Partner Suggestion: one contrarian or non-obvious angle`,

    framework:`Mode: Framework Engine. Visual diagram generated above. Write analysis below.
NEVER use # headings. Bold for section names.

**Framework Logic**
Why this framework fits this problem in 2-3 sentences.

**Branch Analysis**
For each branch: what it covers, data needed, hypothesis tested.

**MECE Check**
- Any gaps or overlaps
- Confirm or flag the structure

**Non-obvious Angle**
One lens most analysts miss.

**Where to Start**
Which branch first and why.`,

    sizing:`Mode: Market Sizing. Visual tree shown above. Write structured narrative below.
NEVER use # headings. NEVER use ## or ###. Bold for section names only.

**Approach and Reasoning**
2-3 sentences on why this segmentation was chosen and the core demand driver.

**Key Assumptions**
- Assumption 1: number and justification
- Assumption 2: number and justification
- Assumption 3: number and justification
- Assumption 4: number and justification
Most sensitive assumption: name the variable that swings the answer most.

**Alternate Reasoning**
- Alternative segmentation approach and why it could work
- Supply-side vs demand-side alternative
- One assumption a different analyst would challenge

**Sanity Check**
Compare to India GDP Rs 300L Cr or population 1.4B. Is the estimate plausible?

**Scenario Analysis**

Bear Case: what goes wrong
- Key variable that drops
- Final estimate: number

Base Case: central scenario
- Final estimate: number from the tree

Bull Case: what goes right
- Key variable that rises
- Final estimate: number

Estimate Range: Bear X - Base Y - Bull Z

**Conclusion and Analysis**
- Who acts on this and why
- What decision this informs
- Single most important insight`,

    financial:`Mode: Financial Analyst.

If the input is vague, ask first:
- What stage is the business — pre-revenue, early, growth, mature?
- What is the primary question — profitability, valuation, unit economics, break-even?
- What time horizon matters — monthly burn, annual P&L, 5-year model?

Otherwise build the analysis:
NEVER use # headings. Bold for section names. Label every number. Show every calculation step.

**Unit Economics**
Revenue per unit, cost per unit, contribution margin — show the math.

**P&L Summary**
Revenue, COGS, gross profit, operating costs, EBITDA — labeled and calculated.

**Key Assumptions**
Flag every assumption that could be wrong and what happens if it is.

**Sensitivity Analysis**
What are the 2 variables that swing the outcome most? Show the range.

**So What**
One-line recommendation based on the numbers.`,

    deck:`Mode: Deck Builder. Read all session outputs. Produce a CEO-ready report by default.
NEVER use # headings. Bold for section names only.

**Situation**
What is happening and why it matters. Context, client, market environment. 2-3 paragraphs using session findings.

**Real Issues**
The 3 most critical problems found. For each:
- Issue name in bold
- What the data shows with specific numbers
- Why it matters to the business
- Root cause in one line

**Recommendations**
3 specific actionable recommendations. For each:
- Recommendation in bold and action-oriented
- Rationale: why this addresses the issue
- Expected impact: what changes and by how much
- Owner and timeline

**Executive Summary**
3 sentences maximum: Situation, Issue, Action.

Only produce PPT slide pointers if the user explicitly asks for slides or PowerPoint.`,
  };
  const ragCtx=rag.length?"\n\nKB:\n"+rag.map(d=>"["+d.source+"] "+d.text).join("\n"):"";
  const histCtx=caseMem.length?"\n\nCASE MEMORY:\n"+caseMem.slice(-2).map(c=>"["+c.title+"] "+c.summary).join("\n"):"";
  const deckCtx=mod==="deck"?buildDeckCtx(sessions):"";
  return "You are CO Consult, your virtual case partner.\nFORMATTING: Never use # ## ### for headings. Use **Bold** for section headings only.\nINPUT SCORE: "+score+"/100\n"+(calGuide[cal]||calGuide.collaborative)+"\n"+(prompts[mod]||"Answer clearly and precisely.")+deckCtx+ragCtx+histCtx;
}

function buildDeckCtx(sessions){
  const parts=["case","framework","sizing","financial"].map(id=>{
    const msgs=sessions[id]||[];
    const ai=msgs.filter(m=>m.role==="assistant"&&m.content&&m.content.length>80&&m.content!=="[FW]"&&m.content!=="[TREE]");
    if(!ai.length)return null;
    return "-- "+id.toUpperCase()+" --\n"+ai.map(m=>m.content).join("\n").slice(0,800);
  }).filter(Boolean);
  if(!parts.length)return "";
  return "\n\nSESSION OUTPUTS (use these as your source):\n"+parts.join("\n\n");
}

async function saveCase(c){try{await window.storage.set("co:"+c.id,JSON.stringify(c));}catch{}}
async function loadCases(){try{const{keys}=await window.storage.list("co:");const cases=[];for(const k of keys){try{const r=await window.storage.get(k);if(r)cases.push(JSON.parse(r.value));}catch{}}return cases.sort((a,b)=>b.ts-a.ts);}catch{return[];}}
async function dropCase(id){try{await window.storage.delete("co:"+id);}catch{}}
function readFile(file){return new Promise(resolve=>{const r=new FileReader();r.onload=e=>resolve(typeof e.target.result==="string"?e.target.result:"");r.onerror=()=>resolve("");r.readAsText(file,"utf-8");});}

// ── SVG layout ─────────────────────────────────────────────────────────────
const NW=148,NH=42,HGAP=56,VGAP=24;
function countLeaves(n){return(!n.children||!n.children.length)?1:n.children.reduce((s,c)=>s+countLeaves(c),0);}
function assignPos(node,depth,yStart,pos){const leaves=countLeaves(node);const totalH=leaves*(NH+VGAP)-VGAP;pos[node.id||"root"]={x:depth*(NW+HGAP),y:yStart+totalH/2-NH/2,w:NW,h:NH};let cy=yStart;(node.children||[]).forEach(c=>{assignPos(c,depth+1,cy,pos);cy+=countLeaves(c)*(NH+VGAP);});return pos;}
function buildEdges(node,pos,edges=[]){const pid=node.id||"root";(node.children||[]).forEach(child=>{const p=pos[pid],c=pos[child.id];if(p&&c)edges.push({x1:p.x+p.w,y1:p.y+p.h/2,x2:c.x,y2:c.y+c.h/2,color:child.color||G_DARK});buildEdges(child,pos,edges);});return edges;}
function flatNodes(node,arr=[]){arr.push(node);(node.children||[]).forEach(c=>flatNodes(c,arr));return arr;}

function FrameworkSVG({data}){
  const root={id:"root",label:data.rootLabel,color:BLACK,children:data.branches};
  const pos={};assignPos(root,0,0,pos);
  const edges=buildEdges(root,pos),allNodes=flatNodes(root);
  const maxX=Math.max(...Object.values(pos).map(p=>p.x+p.w));
  const maxY=Math.max(...Object.values(pos).map(p=>p.y+p.h));
  const pad=18,W=maxX+pad*2,H=maxY+pad*2;
  const path=e=>{const mx=(e.x1+e.x2)/2;return"M "+(e.x1+pad)+" "+(e.y1+pad)+" C "+(mx+pad)+" "+(e.y1+pad)+", "+(mx+pad)+" "+(e.y2+pad)+", "+(e.x2+pad)+" "+(e.y2+pad);};
  return(
    <div style={{overflowX:"auto"}}>
      <svg width={W} height={H} style={{display:"block",fontFamily:"Arial,sans-serif"}}>
        {edges.map((e,i)=><path key={i} d={path(e)} fill="none" stroke={e.color} strokeWidth={1.6} strokeOpacity={0.75}/>)}
        {allNodes.map(node=>{
          const p=pos[node.id||"root"];if(!p)return null;
          const isRoot=(node.id||"root")==="root",isLeaf=!node.children||!node.children.length;
          const bg=isRoot?BLACK:isLeaf?G_LIGHT:(node.color||G_DARK),fg=isRoot?WHITE:isLeaf?G_DARK:WHITE,brd=isRoot?BLACK:(node.color||G_DARK);
          const words=(node.label||"").split(" "),l1=words.slice(0,3).join(" "),l2=words.slice(3).join(" ");
          return(<g key={node.id||"root"}><rect x={p.x+pad} y={p.y+pad} width={p.w} height={p.h} fill={bg} stroke={brd} strokeWidth={1.5} rx={2}/><text x={p.x+pad+p.w/2} y={p.y+pad+p.h/2-(l2?6:0)} textAnchor="middle" dominantBaseline="middle" fontSize={8.5} fontWeight="700" fill={fg}>{l1}</text>{l2&&<text x={p.x+pad+p.w/2} y={p.y+pad+p.h/2+7} textAnchor="middle" dominantBaseline="middle" fontSize={8.5} fontWeight="700" fill={fg}>{l2}</text>}</g>);
        })}
      </svg>
    </div>
  );
}

function OpCircle({op,size=20}){const s={"X":"✕","=":"=","÷":"÷"};return <div style={{width:size,height:size,borderRadius:"50%",background:G_DARK,color:WHITE,display:"flex",alignItems:"center",justifyContent:"center",fontSize:size*0.48,fontWeight:700,flexShrink:0}}>{s[op]||op}</div>;}
function VLine({h=16}){return <div style={{width:1.5,height:h,background:G_DARK,margin:"0 auto",flexShrink:0}}/>;}
function NBox({label,value,sublabel,dark=false,light=false,hl=false,minW=90,px=10,py=5}){
  const bg=hl?G_DARK:dark?G_DARK:light?G_LIGHT:WHITE,col=(dark||hl)?WHITE:BLACK,brd=(dark||hl)?G_DARK:G_BOR;
  return(<div style={{background:bg,border:"1.5px solid "+brd,padding:py+"px "+px+"px",textAlign:"center",minWidth:minW}}><div style={{fontSize:9,fontWeight:700,color:col,lineHeight:1.25}}>{label}</div>{value&&<div style={{fontSize:9,fontWeight:700,color:(dark||hl)?WHITE:G_DARK}}>{value}</div>}{sublabel&&<div style={{fontSize:8,color:(dark||hl)?WHITE:"#555"}}>{sublabel}</div>}</div>);
}
function FilterChain({chain}){return(<div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>{chain.map((item,i)=>(<div key={i} style={{display:"flex",flexDirection:"column",alignItems:"center"}}><VLine/><NBox label={item.label} value={item.value} light minW={110}/>{item.result&&<><VLine/><NBox label={item.result} light minW={110}/></>}</div>))}</div>);}

function GuessTree({data}){
  const{problem,assumptions,tree,formulaBar,answer,difficultyDots,bearCase,bullCase,sensitivityDriver}=data;const dots=difficultyDots||3;
  return(
    <div style={{background:WHITE,border:"1.5px solid "+G_BOR,padding:"16px 14px",fontFamily:"Arial,sans-serif"}}>
      <div style={{borderBottom:"2.5px solid "+G_DARK,paddingBottom:9,marginBottom:11}}>
        <div style={{fontSize:9,fontWeight:700,color:G_MID,letterSpacing:"0.06em",marginBottom:3}}>MARKET SIZING TREE</div>
        <div style={{fontSize:14,fontWeight:900,color:G_DARK,textTransform:"uppercase",lineHeight:1.2}}>{problem}</div>
        <div style={{fontSize:9,color:GRAY2,marginTop:4,display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontWeight:600}}>CO Consult</span>
          <span style={{display:"flex",gap:2}}>{[1,2,3,4,5].map(i=><span key={i} style={{width:13,height:8,border:"1px solid "+G_DARK,background:i<=dots?G_DARK:WHITE,display:"inline-block"}}/>)}</span>
        </div>
      </div>
      <div style={{fontSize:9,fontWeight:700,color:GRAY1,textAlign:"center",marginBottom:9}}>ASSUMPTIONS</div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"4px 16px",marginBottom:14}}>
        {(assumptions||[]).map((a,i)=>(<div key={i} style={{display:"flex",gap:5,fontSize:9,color:GRAY1,lineHeight:1.5}}><span style={{width:13,height:13,borderRadius:"50%",background:G_DARK,color:WHITE,display:"flex",alignItems:"center",justifyContent:"center",fontSize:7.5,fontWeight:700,flexShrink:0}}>{i+1}</span><span>{a}</span></div>))}
      </div>
      <div style={{display:"flex",flexDirection:"column",alignItems:"center"}}>
        <NBox label={tree.root.label} value={tree.root.value} dark minW={160} px={16} py={6}/><VLine h={12}/>
        {tree.level1&&tree.level1.length>0&&(
          <div style={{position:"relative",width:"100%",display:"flex",alignItems:"flex-start",justifyContent:"space-around"}}>
            <div style={{position:"absolute",top:0,left:"8%",right:"8%",height:1.5,background:G_DARK}}/>
            {tree.level1.map((node,i)=>(
              <div key={i} style={{display:"flex",flexDirection:"column",alignItems:"center",flex:1}}>
                <VLine h={12}/><NBox label={node.label} value={node.value} sublabel={node.sublabel} dark minW={90}/>
                {tree.level2&&tree.level2.filter(l2=>l2.parentIndex===i).map((l2g,j)=>(
                  <div key={j} style={{display:"flex",flexDirection:"column",alignItems:"center",width:"100%"}}>
                    <VLine h={10}/>
                    <div style={{position:"relative",width:"95%",display:"flex",alignItems:"flex-start",justifyContent:"space-around"}}>
                      <div style={{position:"absolute",top:0,left:"5%",right:"5%",height:1.5,background:G_DARK}}/>
                      {l2g.nodes.map((n2,k)=>(<div key={k} style={{display:"flex",flexDirection:"column",alignItems:"center",flex:1}}><VLine h={10}/><NBox label={n2.label} value={n2.value} sublabel={n2.sublabel} dark minW={80}/>{tree.filters&&tree.filters.filter(f=>f.nodeLabel===n2.label).map((fc,m)=><FilterChain key={m} chain={fc.chain}/>)}</div>))}
                    </div>
                  </div>
                ))}
                {(!tree.level2||!tree.level2.filter(l2=>l2.parentIndex===i).length)&&tree.filters&&tree.filters.filter(f=>f.nodeLabel===node.label).map((fc,m)=><FilterChain key={m} chain={fc.chain}/>)}
              </div>
            ))}
          </div>
        )}
        {tree.convergence&&<><VLine h={12}/><NBox label={tree.convergence.label} value={tree.convergence.value} light minW={180} px={14} py={5}/></>}
        {tree.calculation&&tree.calculation.length>0&&(
          <><VLine h={12}/>
          <div style={{display:"flex",alignItems:"center",gap:5,flexWrap:"wrap",justifyContent:"center",border:"1.5px solid "+G_BOR,padding:"6px 10px",background:WHITE}}>
            {tree.calculation.map((item,i)=>(<div key={i} style={{display:"flex",alignItems:"center",gap:4}}>{item.operator&&<OpCircle op={item.operator}/>}<NBox label={item.label} value={item.value} hl={item.highlight} light={!item.highlight} minW={80} px={7} py={3}/></div>))}
          </div></>
        )}
      </div>
      {formulaBar&&formulaBar.length>0&&(
        <div style={{display:"flex",alignItems:"center",gap:4,flexWrap:"wrap",marginTop:12,padding:"4px 7px",border:"1px solid "+G_BOR}}>
          {formulaBar.map((item,i)=>(<div key={i} style={{display:"flex",alignItems:"center",gap:4}}><div style={{background:item.isResult?G_DARK:WHITE,border:"1.5px solid "+(item.isResult?G_DARK:G_BOR),padding:"3px 8px",fontSize:8,fontWeight:700,color:item.isResult?WHITE:BLACK,textAlign:"center",minWidth:65}}>{item.label}</div>{item.operator&&<OpCircle op={item.operator} size={15}/>}</div>))}
        </div>
      )}
      <div style={{background:G_RES,border:"1px solid "+G_MID,padding:"6px 12px",marginTop:9,fontSize:10,fontWeight:600,color:BLACK,textAlign:"center",fontStyle:"italic"}}>{answer}</div>
      {(bearCase||bullCase)&&(
        <div style={{display:"flex",gap:8,marginTop:8}}>
          {bearCase&&<div style={{flex:1,background:"#fef2f2",border:"1px solid #fca5a5",padding:"5px 10px",fontSize:9,color:"#991b1b",fontWeight:600,textAlign:"center"}}>🐻 Bear: {bearCase.value}</div>}
          <div style={{flex:1,background:G_LIGHT,border:"1px solid "+G_BOR,padding:"5px 10px",fontSize:9,color:G_DARK,fontWeight:700,textAlign:"center"}}>📊 Base: see estimate above</div>
          {bullCase&&<div style={{flex:1,background:"#f0fdf4",border:"1px solid #86efac",padding:"5px 10px",fontSize:9,color:"#166534",fontWeight:600,textAlign:"center"}}>🐂 Bull: {bullCase.value}</div>}
        </div>
      )}
      {sensitivityDriver&&<div style={{marginTop:7,fontSize:9,color:GRAY2,textAlign:"center"}}>⚡ {sensitivityDriver}</div>}
    </div>
  );
}

function parseBold(text){
  if(!text)return text;
  const parts=[];let last=0;
  const re=/\*\*(.+?)\*\*/g;let m;
  while((m=re.exec(text))!==null){if(m.index>last)parts.push(text.slice(last,m.index));parts.push(<strong key={m.index} style={{color:BLACK,fontWeight:700}}>{m[1]}</strong>);last=m.index+m[0].length;}
  if(last<text.length)parts.push(text.slice(last));
  return parts.length?parts:text;
}

function RichText({text}){
  if(!text)return null;
  return(
    <div>{text.split("\n").map((line,i)=>{
      const t=line.trim();
      if(!t)return <div key={i} style={{height:8}}/>;
      if(/^#{1,3}\s/.test(t)){const label=t.replace(/^#+\s*/,"").replace(/\*\*/g,"");return <div key={i} style={{fontWeight:700,color:G_DARK,fontSize:13,marginTop:16,marginBottom:6,paddingBottom:4,borderBottom:"2px solid "+G_LIGHT}}>{label}</div>;}
      if(t.startsWith("**")&&t.endsWith("**")&&!t.slice(2,-2).includes("**")&&t.length>4){return <div key={i} style={{fontWeight:700,color:G_DARK,fontSize:13,marginTop:16,marginBottom:6,paddingBottom:4,borderBottom:"2px solid "+G_LIGHT}}>{t.slice(2,-2)}</div>;}
      if(/^estimate range/i.test(t)){return <div key={i} style={{background:G_DARK,color:WHITE,padding:"9px 16px",margin:"12px 0",fontWeight:700,fontSize:12.5,textAlign:"center"}}>{t.replace(/\*\*/g,"")}</div>;}
      const isBear=/^bear case/i.test(t)||/^bearish/i.test(t);
      const isBull=/^bull case/i.test(t)||/^bullish/i.test(t);
      const isBase=/^base case/i.test(t);
      if(isBear||isBull||isBase){const emoji=isBear?"🐻":isBull?"🐂":"📊";const bg=isBear?"#fef2f2":isBull?"#f0fdf4":G_LIGHT;const bdr=isBear?"#fca5a5":isBull?"#86efac":G_BOR;const clr=isBear?"#991b1b":isBull?"#166534":G_DARK;return <div key={i} style={{background:bg,border:"1.5px solid "+bdr,padding:"8px 12px",marginTop:8,marginBottom:2,fontSize:12.5,fontWeight:700,color:clr,display:"flex",alignItems:"center",gap:6}}><span style={{fontSize:14,flexShrink:0}}>{emoji}</span><span>{t}</span></div>;}
      if(/^most sensitive/i.test(t)||t.startsWith("⭐")){return <div key={i} style={{background:"#fffbeb",border:"1.5px solid #fcd34d",padding:"7px 12px",margin:"8px 0",fontSize:12,color:"#92400e",fontWeight:600}}>⭐ {t.replace(/^⭐\s*/,"").replace(/^most sensitive assumption:?\s*/i,"")}</div>;}
      if(/^partner suggestion/i.test(t)||t.startsWith("⚡")){return <div key={i} style={{background:G_LIGHT,border:"1.5px solid "+G_BOR,padding:"8px 12px",margin:"8px 0",fontSize:12,color:G_DARK,fontWeight:600}}>⚡ {t.replace(/^⚡\s*/,"").replace(/^partner suggestion:?\s*/i,"")}</div>;}
      if(/^(conclusion|analysis|so what|executive summary)/i.test(t)&&t.includes(":")){const[head,...rest]=t.split(":");return <div key={i} style={{background:G_DARK,color:WHITE,padding:"10px 14px",margin:"12px 0"}}><div style={{fontSize:9,fontWeight:700,letterSpacing:"0.08em",opacity:0.7,marginBottom:4}}>{head.toUpperCase()}</div><div style={{fontSize:12.5,lineHeight:1.65}}>{parseBold(rest.join(":").trim())}</div></div>;}
      if((t.startsWith("-")&&t.length>2)||(t.startsWith("•"))){const c=t.startsWith("•")?t.slice(1).trim():t.slice(2).trim();return <div key={i} style={{display:"flex",gap:8,marginBottom:5,paddingLeft:4,alignItems:"flex-start"}}><span style={{color:G_DARK,fontWeight:700,flexShrink:0,marginTop:3,fontSize:11}}>•</span><span style={{fontSize:12.5,color:GRAY1,lineHeight:1.68}}>{parseBold(c)}</span></div>;}
      if(/^\d+\.\s/.test(t)){const num=t.match(/^\d+/)[0],c=t.replace(/^\d+\.\s*/,"");return <div key={i} style={{display:"flex",gap:9,marginBottom:5,paddingLeft:4,alignItems:"flex-start"}}><span style={{color:WHITE,background:G_DARK,fontWeight:700,flexShrink:0,fontSize:9,width:17,height:17,display:"flex",alignItems:"center",justifyContent:"center",marginTop:2}}>{num}</span><span style={{fontSize:12.5,color:GRAY1,lineHeight:1.68}}>{parseBold(c)}</span></div>;}
      return <div key={i} style={{fontSize:12.5,color:GRAY1,lineHeight:1.72,marginBottom:3}}>{parseBold(line)}</div>;
    })}</div>
  );
}

function ActionBar({content,modId,onRegenerate,onEdit}){
  const [copied,setCopied]=useState(false);
  const [showRedirect,setShowRedirect]=useState(false);
  const [redirectInput,setRedirectInput]=useState("");
  if(!content||content.length<80)return null;
  const clean=content.replace(/\*\*/g,"").replace(/^#{1,3}\s+/gm,"");
  const copy=()=>{navigator.clipboard.writeText(content).catch(()=>{});setCopied(true);setTimeout(()=>setCopied(false),1500);};
  const dlTxt=()=>{const b=new Blob([clean],{type:"text/plain"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="co_consult_output.txt";a.click();URL.revokeObjectURL(u);};
  const dlCsv=()=>{const rows=content.split("\\n").filter(l=>/\d/.test(l)&&l.includes(":"));const csv=rows.map(l=>l.replace(/\*\*/g,"").replace(/:/,",")).join("\\n");const b=new Blob([csv||clean],{type:"text/csv"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="co_consult_data.csv";a.click();URL.revokeObjectURL(u);};
  return(
    <div style={{marginTop:8}}>
      <div style={{display:"flex",gap:5,flexWrap:"wrap",alignItems:"center"}}>
        {onRegenerate&&<button onClick={onRegenerate} style={{padding:"4px 11px",background:WHITE,border:"1px solid "+G_BOR,color:G_DARK,fontSize:9,fontWeight:600,cursor:"pointer",fontFamily:"Arial,sans-serif",display:"flex",alignItems:"center",gap:4}}>↺ Regenerate</button>}
        <button onClick={()=>setShowRedirect(r=>!r)} style={{padding:"4px 11px",background:showRedirect?G_DARK:WHITE,border:"1px solid "+G_BOR,color:showRedirect?WHITE:G_DARK,fontSize:9,fontWeight:600,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>✎ Change this</button>
        <button onClick={copy} style={{padding:"4px 11px",background:WHITE,border:"1px solid "+G_BOR,color:GRAY2,fontSize:9,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>{copied?"✓ Copied":"Copy"}</button>
        <button onClick={dlTxt} style={{padding:"4px 11px",background:WHITE,border:"1px solid "+G_BOR,color:GRAY2,fontSize:9,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>↓ Word</button>
        {modId==="financial"&&<button onClick={dlCsv} style={{padding:"4px 11px",background:WHITE,border:"1px solid "+G_BOR,color:GRAY2,fontSize:9,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>↓ Excel</button>}
        {modId==="deck"&&<button onClick={()=>{}} style={{padding:"4px 11px",background:G_DARK,border:"none",color:WHITE,fontSize:9,fontWeight:600,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>↓ PPT</button>}
      </div>
      {showRedirect&&(
        <div style={{marginTop:7,display:"flex",gap:6,alignItems:"flex-end"}}>
          <input
            value={redirectInput}
            onChange={e=>setRedirectInput(e.target.value)}
            onKeyDown={e=>{if(e.key==="Enter"&&redirectInput.trim()){onEdit&&onEdit(redirectInput);setRedirectInput("");setShowRedirect(false);}}}
            placeholder="Tell me how to change this — e.g. make it shorter, focus on cost side, add a risk section..."
            style={{flex:1,padding:"7px 11px",border:"1.5px solid "+G_BOR,background:G_PALE,fontSize:12,fontFamily:"Arial,sans-serif",color:BLACK,outline:"none"}}
            autoFocus
          />
          <button onClick={()=>{if(redirectInput.trim()){onEdit&&onEdit(redirectInput);setRedirectInput("");setShowRedirect(false);}}}
            style={{padding:"7px 14px",background:G_DARK,border:"none",color:WHITE,fontSize:11,fontWeight:700,cursor:"pointer",fontFamily:"Arial,sans-serif",whiteSpace:"nowrap"}}>
            Apply
          </button>
        </div>
      )}
    </div>
  );
}

function ScoreBadge({score,cal}){
  const cols={directive:"#c0392b",guided:"#e67e22",collaborative:G_MID,challenging:"#2980b9"};
  const c=cols[cal]||GRAY3;
  return <div style={{display:"inline-flex",alignItems:"center",gap:4,background:c+"18",border:"1px solid "+c+"35",padding:"2px 8px",fontSize:8.5}}><div style={{width:4,height:4,borderRadius:"50%",background:c}}/><span style={{color:c,fontWeight:700}}>{(cal||"").toUpperCase()} {score}/100</span></div>;
}

function Msg({msg,streaming,onRegenerate,onEdit}){
  const isUser=msg.role==="user";
  if(msg.role==="system")return <div style={{textAlign:"center",margin:"5px 0",fontSize:9,color:GRAY3}}>--- {msg.content} ---</div>;
  const hasContent=msg.content&&msg.content!=="[FW]"&&msg.content!=="[TREE]";
  return(
    <div style={{display:"flex",gap:9,marginBottom:14,flexDirection:isUser?"row-reverse":"row",animation:"fadein 0.18s ease"}}>
      <div style={{width:24,height:24,flexShrink:0,border:"1.5px solid "+(isUser?GRAY4:G_DARK),background:isUser?G_PALE:G_DARK,display:"flex",alignItems:"center",justifyContent:"center",fontSize:8,fontWeight:700,color:isUser?G_DARK:WHITE}}>{isUser?"YOU":"CC"}</div>
      <div style={{flex:1,maxWidth:msg.frameworkData||msg.treeData?"100%":"84%",display:"flex",flexDirection:"column",gap:6,alignItems:isUser?"flex-end":"flex-start",width:msg.frameworkData||msg.treeData?"100%":"auto"}}>
        {!isUser&&msg.score!==undefined&&<ScoreBadge score={msg.score} cal={msg.calibration}/>}
        {msg.frameworkData&&(
          <div style={{width:"100%"}}>
            <div style={{fontSize:9,fontWeight:700,color:G_DARK,marginBottom:6}}>MECE FRAMEWORK — {msg.frameworkData.frameworkName}</div>
            <div style={{background:WHITE,border:"1.5px solid "+G_DARK,padding:"14px 12px",marginBottom:8}}>
              <div style={{fontSize:13,fontWeight:900,color:G_DARK,marginBottom:10}}>{msg.frameworkData.title}</div>
              <FrameworkSVG data={msg.frameworkData}/>
            </div>
            <div style={{display:"flex",gap:7,flexWrap:"wrap"}}>
              {[{label:"MECE CHECK",content:msg.frameworkData.meceCheck,color:G_MID},{label:"KEY HYPOTHESIS",content:msg.frameworkData.keyHypothesis,color:"#c2410c"},{label:"NON-OBVIOUS ANGLE",content:msg.frameworkData.nonObviousAngle,color:"#2980b9"}].map((card,ci)=>(
                <div key={ci} style={{flex:1,minWidth:150,background:WHITE,border:"1px solid "+G_BOR,padding:"8px 10px"}}><div style={{fontSize:8.5,fontWeight:700,color:card.color,marginBottom:4}}>{card.label}</div><div style={{fontSize:11,color:GRAY1,lineHeight:1.55}}>{card.content}</div></div>
              ))}
            </div>
          </div>
        )}
        {msg.treeData&&<div style={{width:"100%"}}><GuessTree data={msg.treeData}/></div>}
        {hasContent&&(
          <div style={{width:isUser?"auto":"100%"}}>
            <div style={{background:isUser?G_DARK:WHITE,border:"1.5px solid "+(isUser?G_DARK:G_BOR),padding:"10px 14px",fontSize:13,lineHeight:1.72,color:isUser?WHITE:BLACK,wordBreak:"break-word"}}>
              {streaming&&!msg.content?<span style={{color:G_MID,fontStyle:"italic"}}>thinking...</span>:isUser?msg.content:<RichText text={msg.content}/>}
              {streaming&&msg.content&&<span style={{color:G_MID,animation:"blink 0.7s infinite"}}>|</span>}
            </div>
            {!isUser&&!streaming&&msg.content&&msg.content.length>100&&(
              <ActionBar
                content={msg.content}
                modId={msg.modId}
                onRegenerate={()=>onRegenerate && onRegenerate(msg)}
                onEdit={()=>onEdit && onEdit(msg)}
              />
            )}
          </div>
        )}
        {streaming&&!msg.content&&!msg.frameworkData&&!msg.treeData&&<div style={{background:WHITE,border:"1.5px solid "+G_BOR,padding:"10px 14px",fontSize:13,color:G_MID,fontStyle:"italic"}}>thinking...</div>}
      </div>
    </div>
  );
}

function DeckReady({sessions}){
  const checks=[{id:"case",label:"Case scoped"},{id:"framework",label:"Framework built"},{id:"sizing",label:"Market sized"},{id:"financial",label:"Financials modelled"}];
  const done=checks.filter(c=>sessions[c.id]?.length>0),pct=Math.round(done.length/4*100);
  return(
    <div style={{background:WHITE,border:"1.5px solid "+G_BOR,padding:"12px 14px",marginBottom:14}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}><div style={{fontSize:10,fontWeight:700,color:G_DARK}}>Deck Readiness</div><div style={{fontSize:10,fontWeight:700,color:G_DARK}}>{pct}%</div></div>
      <div style={{height:3,background:G_LIGHT,marginBottom:8,overflow:"hidden"}}><div style={{height:"100%",width:pct+"%",background:G_DARK,transition:"width 0.4s"}}/></div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}}>{checks.map(c=>{const isDone=sessions[c.id]?.length>0;return(<div key={c.id} style={{display:"flex",alignItems:"center",gap:5,fontSize:11,color:isDone?BLACK:GRAY3}}><span style={{color:isDone?G_DARK:GRAY4,fontSize:13,fontWeight:700}}>{isDone?"✓":"○"}</span>{c.label}</div>);})}</div>
      {pct===100&&<div style={{marginTop:8,padding:"5px 8px",background:G_LIGHT,border:"1px solid "+G_BOR,fontSize:10,color:G_DARK,fontWeight:600}}>All modules complete. Type anything to generate the deck.</div>}
    </div>
  );
}

function MemoryModal({cases,onSelect,onDelete,onClose}){
  return(
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.45)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:200}} onClick={onClose}>
      <div onClick={e=>e.stopPropagation()} style={{width:520,maxHeight:"72vh",background:WHITE,border:"2px solid "+G_DARK,display:"flex",flexDirection:"column",overflow:"hidden"}}>
        <div style={{padding:"12px 16px",borderBottom:"1.5px solid "+G_BOR,display:"flex",justifyContent:"space-between",alignItems:"center",background:G_PALE}}>
          <div><div style={{fontSize:13,fontWeight:700,color:G_DARK}}>Case Memory</div><div style={{fontSize:9,color:GRAY2,marginTop:1}}>{cases.length}/{MAX_CASES} slots</div></div>
          <button onClick={onClose} style={{background:"none",border:"none",color:GRAY2,cursor:"pointer",fontSize:18}}>✕</button>
        </div>
        <div style={{flex:1,overflowY:"auto",padding:"10px 12px"}}>
          {cases.length===0?<div style={{textAlign:"center",padding:"24px 0",color:GRAY3,fontSize:11}}>No saved cases yet</div>:cases.map(c=>(
            <div key={c.id} style={{background:G_PALE,border:"1px solid "+G_BOR,padding:"9px 11px",marginBottom:6,display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
              <div style={{flex:1,cursor:"pointer"}} onClick={()=>onSelect(c)}>
                <div style={{fontSize:12,color:G_DARK,marginBottom:2,fontWeight:700}}>{c.title}</div>
                <div style={{fontSize:9.5,color:GRAY2}}>{c.summary}</div>
                <div style={{fontSize:8.5,color:GRAY3,marginTop:2}}>{c.module} · {new Date(c.ts).toLocaleDateString()}</div>
              </div>
              <button onClick={()=>onDelete(c.id)} style={{background:"none",border:"none",color:GRAY3,cursor:"pointer",fontSize:14,marginLeft:6}} onMouseEnter={e=>e.target.style.color="#c0392b"} onMouseLeave={e=>e.target.style.color=GRAY3}>✕</button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function KBModal({kb,kbFiles,setKbFiles,onClose}){
  const [log,setLog]=useState([]);const [busy,setBusy]=useState(false);const fRef=useRef();
  const process=async files=>{setBusy(true);for(const f of files){setLog(l=>[...l,"Loading "+f.name+"..."]);try{const text=await readFile(f);kb.add(f.name,text&&text.length>50?text:"Document: "+f.name);setKbFiles(p=>[...new Set([...p,f.name])]);setLog(l=>[...l,"Done: "+f.name]);}catch(e){setLog(l=>[...l,"Error: "+f.name]);}}setBusy(false);};
  return(
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.45)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:200}} onClick={onClose}>
      <div onClick={e=>e.stopPropagation()} style={{width:420,background:WHITE,border:"2px solid "+G_DARK,padding:"18px 20px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:12}}>
          <div><div style={{fontSize:13,fontWeight:700,color:G_DARK}}>Knowledge Base</div><div style={{fontSize:9.5,color:GRAY2,marginTop:1}}>Upload cases and frameworks</div></div>
          <button onClick={onClose} style={{background:"none",border:"none",color:GRAY2,cursor:"pointer",fontSize:18}}>✕</button>
        </div>
        <div onClick={()=>fRef.current?.click()} style={{border:"1.5px dashed "+G_BOR,padding:"20px",textAlign:"center",cursor:"pointer",marginBottom:10}}>
          <div style={{fontSize:11,color:GRAY2}}>Click to browse files</div>
          <div style={{fontSize:9,color:GRAY3,marginTop:2}}>PDF · TXT · MD · DOC</div>
        </div>
        <input ref={fRef} type="file" multiple accept=".pdf,.txt,.md,.doc,.docx" style={{display:"none"}} onChange={e=>process([...e.target.files])}/>
        {log.length>0&&<div style={{background:G_PALE,border:"1px solid "+G_BOR,padding:"7px 9px",maxHeight:90,overflowY:"auto",fontSize:9,lineHeight:1.8}}>{log.map((l,i)=><div key={i}>{l}</div>)}{busy&&<div style={{color:G_MID}}>indexing...</div>}</div>}
        <div style={{marginTop:10,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div style={{fontSize:9.5,color:GRAY2}}>{kb.count} chunks · {kbFiles.length} files</div>
          <button onClick={onClose} style={{padding:"5px 16px",background:G_DARK,border:"none",color:WHITE,fontSize:11,fontWeight:700,cursor:"pointer"}}>Done</button>
        </div>
      </div>
    </div>
  );
}

function Intro({onDone}){
  const [phase,setPhase]=useState(0);
  // phase 0: blank → logo fades in
  // phase 1: tagline types out
  // phase 2: capabilities appear
  // phase 3: CTA appears
  // phase 4: done (fade out)
  const tagline="Hi, I'm your virtual consulting partner.";
  const [typed,setTyped]=useState("");
  const [fading,setFading]=useState(false);

  useEffect(()=>{
    const t1=setTimeout(()=>setPhase(1),600);
    return()=>clearTimeout(t1);
  },[]);

  useEffect(()=>{
    if(phase!==1)return;
    let i=0;
    const iv=setInterval(()=>{
      i++;
      setTyped(tagline.slice(0,i));
      if(i>=tagline.length){clearInterval(iv);setTimeout(()=>setPhase(2),400);}
    },38);
    return()=>clearInterval(iv);
  },[phase]);

  useEffect(()=>{
    if(phase===2)setTimeout(()=>setPhase(3),700);
  },[phase]);

  const caps=[
    {icon:"◈",text:"Case Scoping & MECE Frameworks"},
    {icon:"∑",text:"Market Sizing & Guesstimates"},
    {icon:"⌥",text:"Financial Analysis & Unit Economics"},
    {icon:"▤",text:"CEO Reports & Deck Synthesis"},
  ];

  const handleStart=()=>{
    setFading(true);
    setTimeout(()=>onDone(),500);
  };

  return(
    <div style={{position:"fixed",inset:0,background:"#f5f5f0",display:"flex",alignItems:"center",justifyContent:"center",zIndex:300,fontFamily:"Arial,sans-serif",opacity:fading?0:1,transition:"opacity 0.5s ease"}}>
      <style>{`
        @keyframes fadeup{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
        @keyframes fadein2{from{opacity:0}to{opacity:1}}
        @keyframes blink2{0%,100%{opacity:1}50%{opacity:0}}
        @keyframes pulse2{0%,100%{transform:scale(1)}50%{transform:scale(1.04)}}
        .cap-item{animation:fadeup 0.5s ease forwards;opacity:0;}
        .cta-btn:hover{background:#1a3d14!important;transform:translateY(-1px);}
      `}</style>
      <div style={{width:"100%",maxWidth:520,padding:"0 28px",textAlign:"center"}}>

        {/* Logo — always visible once mounted */}
        <div style={{marginBottom:32,animation:"fadein2 0.6s ease"}}>
          <div style={{display:"inline-flex",alignItems:"center",gap:12,marginBottom:16}}>
            <div style={{lineHeight:0.9}}>
              <div style={{fontSize:48,fontWeight:900,color:G_DARK,letterSpacing:"-0.05em",lineHeight:1}}>CO</div>
              <div style={{fontSize:10,fontWeight:700,color:G_MID,letterSpacing:"0.2em",marginTop:2}}>CONSULT</div>
            </div>
            <div style={{width:2,height:52,background:G_BOR}}/>
            <div style={{textAlign:"left"}}>
              <div style={{fontSize:11,color:G_BOR,letterSpacing:"0.08em",fontWeight:600,marginBottom:3}}>POWERED BY CLAUDE</div>
              <div style={{fontSize:13,color:GRAY2,fontWeight:400,lineHeight:1.5}}>Strategy · Finance<br/>Frameworks · Insights</div>
            </div>
          </div>

          {/* Thin green line */}
          <div style={{width:48,height:2,background:G_DARK,margin:"0 auto"}}/>
        </div>

        {/* Typewriter tagline */}
        <div style={{fontSize:22,fontWeight:600,color:G_DARK,marginBottom:28,minHeight:30,letterSpacing:"-0.01em"}}>
          {typed}
          {phase===1&&<span style={{animation:"blink2 0.7s infinite",color:G_MID}}>|</span>}
        </div>

        {/* Capabilities */}
        {phase>=2&&(
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:32}}>
            {caps.map((c,i)=>(
              <div key={i} className="cap-item" style={{animationDelay:(i*120)+"ms",background:WHITE,border:"1.5px solid "+G_BOR,padding:"10px 14px",display:"flex",alignItems:"center",gap:9,textAlign:"left"}}>
                <span style={{fontSize:16,color:G_DARK,flexShrink:0}}>{c.icon}</span>
                <span style={{fontSize:11.5,color:GRAY1,fontWeight:500,lineHeight:1.4}}>{c.text}</span>
              </div>
            ))}
          </div>
        )}

        {/* CTA */}
        {phase>=3&&(
          <div style={{animation:"fadeup 0.5s ease"}}>
            <button className="cta-btn" onClick={handleStart} style={{padding:"13px 40px",background:G_DARK,border:"none",color:WHITE,fontSize:13,fontWeight:700,cursor:"pointer",letterSpacing:"0.06em",transition:"all 0.2s ease",fontFamily:"Arial,sans-serif"}}>
              GET STARTED →
            </button>
            <div style={{marginTop:12,fontSize:10,color:GRAY3}}>Case Scoping · Market Sizing · Frameworks · Deck Builder</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App(){
  const kb=useMemo(()=>{const k=new KB();k.add("seed",SEED);return k;},[]);
  const [kbFiles,setKbFiles]=useState(["seed"]);
  const [activeMod,setActiveMod]=useState("case");
  const [sessions,setSessions]=useState(Object.fromEntries(MODS.map(m=>[m.id,[]])));
  const [input,setInput]=useState("");
  const [loading,setLoading]=useState(false);
  const [scoring,setScoring]=useState(false);
  const [treeLoading,setTreeLoading]=useState(false);
  const [fwLoading,setFwLoading]=useState(false);
  const [showMemory,setShowMemory]=useState(false);
  const [showKB,setShowKB]=useState(false);
  const [showHistory,setShowHistory]=useState(false);
  const [savedCases,setSavedCases]=useState([]);
  const [sideOpen,setSideOpen]=useState(true);
  const [ready,setReady]=useState(false);
  const endRef=useRef(null),taRef=useRef(null),fileRef=useRef();
  const msgs=sessions[activeMod],mod=MODS.find(m=>m.id===activeMod);

  useEffect(()=>{loadCases().then(setSavedCases);},[]);
  useEffect(()=>{endRef.current?.scrollIntoView({behavior:"smooth"});},[msgs,loading,treeLoading,fwLoading]);
  useEffect(()=>{if(taRef.current){taRef.current.style.height="auto";taRef.current.style.height=Math.min(taRef.current.scrollHeight,120)+"px";}},[input]);

  const persistCase=useCallback(async(history,modId,firstMsg)=>{
    if(savedCases.length>=MAX_CASES)return;
    const c={id:modId+"_"+Date.now(),module:modId,title:firstMsg.slice(0,55),summary:firstMsg.slice(0,140),turns:Math.floor(history.length/2),ts:Date.now(),history};
    await saveCase(c);setSavedCases(p=>[c,...p.filter(x=>x.id!==c.id)].slice(0,MAX_CASES));
  },[savedCases.length]);



  const send=useCallback(async(text,attached="")=>{
    if((!text.trim()&&!attached)||loading)return;
    const full=attached?text+"\n\n[ATTACHED]\n"+attached.slice(0,2800):text;
    const userMsg={role:"user",content:full};
    const history=[...msgs,userMsg];
    setSessions(p=>({...p,[activeMod]:history}));
    setInput("");setLoading(true);

    const genFW=async()=>{
      setFwLoading(true);let fw=null;
      try{const raw=await callJSON(full,FW_SYSTEM);const j=raw.match(/\{[\s\S]*\}/)?.[0];if(j)fw=JSON.parse(j);}catch(e){console.error(e);}
      setFwLoading(false);return fw;
    };

    if(activeMod==="case"||activeMod==="framework"){
      const fw=await genFW();
      if(fw){
        const fwMsg={role:"assistant",content:"[FW]",frameworkData:fw,score:80,calibration:"collaborative"};
        const narIdx=history.length+1;
        const narMsg={role:"assistant",content:"",score:80,calibration:"collaborative",modId:activeMod};
        setSessions(p=>({...p,[activeMod]:[...history,fwMsg,narMsg]}));
        try{
          const sys=buildSystem(activeMod,"collaborative",80,kb.retrieve(full,4),savedCases,sessions);
          let nar="";
          await streamClaude(history.map(m=>({role:m.role,content:m.content})),sys,chunk=>{
            nar+=chunk;
            setSessions(p=>{const u=[...p[activeMod]];if(u[narIdx])u[narIdx]={...u[narIdx],content:nar};return{...p,[activeMod]:u};});
          });
        }catch(e){console.error(e);}
        if(history.length===1)persistCase([...history,fwMsg],activeMod,text);
        setLoading(false);return;
      }
    }

    if(activeMod==="sizing"){
      setTreeLoading(true);let tree=null;
      try{const raw=await callJSON(text,TREE_SYSTEM);const j=raw.match(/\{[\s\S]*\}/)?.[0];if(j)tree=JSON.parse(j);}catch(e){console.error(e);}
      setTreeLoading(false);
      if(tree){
        const treeMsg={role:"assistant",content:"[TREE]",treeData:tree,score:80,calibration:"collaborative"};
        const narIdx=history.length+1;
        const narMsg={role:"assistant",content:"",score:80,calibration:"collaborative",modId:"sizing"};
        setSessions(p=>({...p,[activeMod]:[...history,treeMsg,narMsg]}));
        try{
          const sys=buildSystem("sizing","collaborative",80,kb.retrieve(text,4),savedCases,sessions);
          let nar="";
          await streamClaude(history.map(m=>({role:m.role,content:m.content})),sys,chunk=>{
            nar+=chunk;
            setSessions(p=>{const u=[...p[activeMod]];if(u[narIdx])u[narIdx]={...u[narIdx],content:nar};return{...p,[activeMod]:u};});
          });
        }catch(e){console.error(e);}
        if(history.length===1)persistCase([...history,treeMsg],activeMod,text);
        setLoading(false);return;
      }
    }

    if(activeMod==="deck"){
      const su=(sessions["sizing"]||[]).find(m=>m.role==="user");
      if(su?.content){
        setTreeLoading(true);let dt=null;
        try{const raw=await callJSON(su.content,TREE_SYSTEM);const j=raw.match(/\{[\s\S]*\}/)?.[0];if(j)dt=JSON.parse(j);}catch{}
        setTreeLoading(false);
        if(dt){
          const treeMsg={role:"assistant",content:"[TREE]",treeData:dt,score:80,calibration:"collaborative"};
          const narIdx=history.length+1;
          const narMsg={role:"assistant",content:"",score:80,calibration:"collaborative",modId:"deck"};
          setSessions(p=>({...p,[activeMod]:[...history,treeMsg,narMsg]}));
          try{
            const sys=buildSystem("deck","collaborative",80,kb.retrieve(full,4),savedCases,sessions);
            let resp="";
            await streamClaude(history.map(m=>({role:m.role,content:m.content})),sys,chunk=>{
              resp+=chunk;
              setSessions(p=>{const u=[...p[activeMod]];if(u[narIdx])u[narIdx]={...u[narIdx],content:resp};return{...p,[activeMod]:u};});
            });
            if(history.length===1)persistCase([...history,treeMsg],activeMod,text);
          }catch(e){console.error(e);}
          setLoading(false);return;
        }
      }
    }

    let meta={score:50,calibration:"guided",gaps:[],strengths:[]};
    setScoring(true);try{meta=await scoreInput(full,activeMod);}catch{}setScoring(false);
    const aMsg={role:"assistant",content:"",score:meta.score,calibration:meta.calibration,modId:activeMod};
    const aMsgIdx=history.length;
    setSessions(p=>({...p,[activeMod]:[...history,aMsg]}));
    try{
      const sys=buildSystem(activeMod,meta.calibration,meta.score,kb.retrieve(full,4),savedCases,sessions);
      let resp="";
      await streamClaude(history.map(m=>({role:m.role,content:m.content})),sys,chunk=>{
        resp+=chunk;
        setSessions(p=>{const u=[...p[activeMod]];if(u[aMsgIdx])u[aMsgIdx]={...u[aMsgIdx],content:resp};return{...p,[activeMod]:u};});
      });
      if(history.length===1)persistCase([...history,{role:"assistant",content:resp,modId:activeMod}],activeMod,text);
    }catch(err){
      setSessions(p=>{const u=[...p[activeMod]];if(u[aMsgIdx])u[aMsgIdx]={...u[aMsgIdx],content:"Error: "+err.message};return{...p,[activeMod]:u};});
    }
    setLoading(false);
  },[msgs,loading,activeMod,kb,savedCases,persistCase,sessions]);

  const handleKey=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send(input);}};
  const handleAttach=async e=>{const f=e.target.files[0];if(!f)return;const t=await readFile(f);send(input||"Analyse: "+f.name,t);e.target.value="";};
  const isWorking=loading||scoring||treeLoading||fwLoading;
  const hasAnySessions=MODS.some(m=>sessions[m.id]?.length>0);

  if(!ready)return <Intro onDone={()=>setReady(true)}/>;

  return(
    <div style={{display:"flex",flexDirection:"column",height:"100vh",background:"#f5f5f0",color:BLACK,fontFamily:"Arial,sans-serif",overflow:"hidden"}}>
      <style>{`*{box-sizing:border-box;margin:0;padding:0;}::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-thumb{background:#5a8a4f;}textarea:focus{outline:none;}@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:translateY(0)}}@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}@keyframes spin{to{transform:rotate(360deg)}}@keyframes pulse{0%,100%{opacity:0.6}50%{opacity:1}}.nb:hover{background:#e8f5e0!important;color:#2d5a27!important;}.qb:hover{border-color:#2d5a27!important;color:#2d5a27!important;background:#e8f5e0!important;}.tb:hover{background:#e8f5e0!important;border-color:#2d5a27!important;color:#2d5a27!important;}`}</style>

      {/* TOPBAR */}
      <div style={{height:48,background:WHITE,borderBottom:"2px solid "+G_DARK,display:"flex",alignItems:"center",padding:"0 16px",flexShrink:0,gap:0}}>
        {/* Left — logo */}
        <div style={{display:"flex",alignItems:"center",gap:10,flexShrink:0}}>
          <div style={{lineHeight:1}}><div style={{fontSize:20,fontWeight:900,color:G_DARK,letterSpacing:"-0.04em"}}>CO</div><div style={{fontSize:7.5,fontWeight:700,color:G_MID,letterSpacing:"0.14em"}}>CONSULT</div></div>
          <div style={{width:1.5,height:28,background:G_BOR}}/>
          <div style={{lineHeight:1.4}}><div style={{fontSize:11.5,fontWeight:600,color:G_DARK}}>Your virtual case partner</div><div style={{fontSize:9,color:G_MID}}>Strategy · Finance · Frameworks</div></div>
        </div>
        {/* Centre — session pills */}
        <div style={{flex:1,display:"flex",justifyContent:"center",alignItems:"center",gap:8,padding:"0 16px"}}>
          {isWorking?(
            <div style={{display:"flex",alignItems:"center",gap:6,background:G_LIGHT,border:"1px solid "+G_BOR,padding:"4px 12px"}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:G_DARK,animation:"pulse 1s infinite"}}/>
              <span style={{fontSize:9.5,color:G_DARK,fontWeight:600}}>{fwLoading?"Building framework...":treeLoading?"Building tree...":scoring?"Assessing...":"Thinking..."}</span>
            </div>
          ):hasAnySessions?(
            MODS.map(m=>{const has=sessions[m.id]?.length>0;return has?(
              <div key={m.id} onClick={()=>setActiveMod(m.id)} style={{display:"flex",alignItems:"center",gap:4,padding:"3px 9px",background:activeMod===m.id?G_DARK:G_LIGHT,border:"1px solid "+G_BOR,cursor:"pointer",transition:"all 0.1s"}}>
                <span style={{fontSize:11,color:activeMod===m.id?WHITE:G_DARK}}>{m.icon}</span>
                <span style={{fontSize:8.5,fontWeight:600,color:activeMod===m.id?WHITE:G_DARK}}>{m.label.split(" ")[0]}</span>
              </div>
            ):null;})
          ):(
            <span style={{fontSize:10,color:GRAY3,fontStyle:"italic"}}>Start a new case below</span>
          )}
        </div>
        {/* Right — controls */}
        <div style={{display:"flex",gap:5,alignItems:"center",flexShrink:0}}>
          <button className="tb" onClick={()=>setShowKB(true)} style={{padding:"4px 9px",background:WHITE,border:"1px solid "+G_BOR,color:kbFiles.length>1?G_DARK:GRAY3,fontSize:9,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>Upload</button>
          <button className="tb" onClick={()=>setShowMemory(true)} style={{padding:"4px 9px",background:WHITE,border:"1px solid "+G_BOR,color:savedCases.length?G_DARK:GRAY3,fontSize:9,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>Memory</button>
          <button className="tb" onClick={()=>setSessions(p=>({...p,[activeMod]:[]}))} style={{padding:"4px 9px",background:WHITE,border:"1px solid "+G_BOR,color:GRAY3,fontSize:9,cursor:"pointer",fontFamily:"Arial,sans-serif"}}>Clear</button>
        </div>
      </div>

      {/* BODY */}
      <div style={{flex:1,display:"flex",overflow:"hidden"}}>
        {/* SIDEBAR */}
        <div style={{width:sideOpen?182:40,background:WHITE,borderRight:"1.5px solid "+G_BOR,display:"flex",flexDirection:"column",transition:"width 0.2s",flexShrink:0,overflow:"hidden"}}>
          {/* Sidebar header */}
          <div style={{padding:"7px 8px",borderBottom:"1px solid "+G_LIGHT,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            {sideOpen&&<span style={{fontSize:9,fontWeight:700,color:G_MID,letterSpacing:"0.1em"}}>TOOLS</span>}
            <button onClick={()=>setSideOpen(o=>!o)} style={{background:"none",border:"none",color:G_MID,cursor:"pointer",fontSize:11,padding:0,fontWeight:700,marginLeft:"auto"}}>{sideOpen?"<":">"}</button>
          </div>
          {/* Module list */}
          <div style={{flex:1,padding:"5px 5px 0",overflowY:"auto"}}>
            {MODS.map(m=>{const active=activeMod===m.id,has=sessions[m.id]?.length>0;return(
              <button key={m.id} className="nb" onClick={()=>setActiveMod(m.id)} style={{width:"100%",padding:sideOpen?"8px 10px":"9px 7px",background:active?G_LIGHT:WHITE,border:"none",borderLeft:active?"3px solid "+G_DARK:"3px solid transparent",color:active?G_DARK:has?GRAY1:GRAY2,fontSize:11.5,cursor:"pointer",textAlign:"left",marginBottom:1,display:"flex",alignItems:"center",gap:8,transition:"all 0.1s",fontFamily:"Arial,sans-serif",fontWeight:active?700:400}}>
                <span style={{flexShrink:0,fontSize:13}}>{m.icon}</span>
                {sideOpen&&<span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{m.label}</span>}
                {sideOpen&&has&&<span style={{marginLeft:"auto",width:5,height:5,borderRadius:"50%",background:active?G_DARK:G_BOR,flexShrink:0}}/>}
              </button>
            );})}
          </div>
          {/* Case History */}
          {sideOpen&&(
            <div style={{borderTop:"1px solid "+G_LIGHT,padding:"6px 7px"}}>
              <button onClick={()=>setShowHistory(h=>!h)} style={{width:"100%",display:"flex",alignItems:"center",justifyContent:"space-between",background:"none",border:"none",cursor:"pointer",padding:"3px 3px",fontFamily:"Arial,sans-serif",marginBottom:4}}>
                <span style={{fontSize:9,fontWeight:700,color:G_MID,letterSpacing:"0.08em"}}>HISTORY {savedCases.length>0?"("+savedCases.length+")":""}</span>
                <span style={{fontSize:10,color:G_MID}}>{showHistory?"▴":"▾"}</span>
              </button>
              {showHistory&&savedCases.length>0&&(
                <div style={{maxHeight:190,overflowY:"auto"}}>
                  {savedCases.slice(0,10).map(c=>(
                    <button key={c.id} onClick={()=>{setActiveMod(c.module);setSessions(p=>({...p,[c.module]:c.history}));}} style={{width:"100%",padding:"5px 5px",background:"none",border:"none",cursor:"pointer",textAlign:"left",marginBottom:1,fontFamily:"Arial,sans-serif",borderRadius:2}}
                      onMouseEnter={e=>e.currentTarget.style.background=G_LIGHT}
                      onMouseLeave={e=>e.currentTarget.style.background="none"}>
                      <div style={{fontSize:9.5,color:G_DARK,fontWeight:600,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{c.title}</div>
                      <div style={{fontSize:8,color:GRAY3,marginTop:1}}>{c.module} · {new Date(c.ts).toLocaleDateString()}</div>
                    </button>
                  ))}
                </div>
              )}
              {showHistory&&savedCases.length===0&&<div style={{fontSize:9,color:GRAY3,padding:"4px 3px"}}>No history yet</div>}
            </div>
          )}
        </div>

        {/* MAIN */}
        <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>
          <div style={{flex:1,overflowY:"auto",padding:"16px 18px",background:"#f5f5f0"}}>
            {msgs.length===0?(
              <div style={{maxWidth:560,margin:"16px auto 0",animation:"fadein 0.3s ease"}}>
                <div style={{background:WHITE,border:"1.5px solid "+G_DARK,padding:"14px 16px",marginBottom:12}}>
                  <div style={{fontSize:14,fontWeight:700,color:G_DARK,marginBottom:5}}>{mod.label}</div>
                  <div style={{fontSize:12.5,color:GRAY1,lineHeight:1.7}}>
                    {activeMod==="case"&&"Describe your client problem. A MECE framework diagram generates automatically, then the narrative analysis follows."}
                    {activeMod==="framework"&&"Describe the problem and get a visual MECE diagram with the consulting analysis beneath it."}
                    {activeMod==="sizing"&&"Type any market sizing or guestimate problem. A structured visual tree generates with Bear, Base, and Bull scenarios."}
                    {activeMod==="financial"&&"Unit economics, break-even, valuation — every number labeled, every assumption flagged. Export to Excel or Word."}
                    {activeMod==="deck"&&(hasAnySessions?"Session work detected. The deck synthesises all findings into a CEO report — Situation, Issues, Recommendations.":"Complete other modules first, or paste your existing analysis here to build the deck.")}
                  </div>
                </div>
                {activeMod==="deck"&&<DeckReady sessions={sessions}/>}
                <div style={{textAlign:"center",padding:"22px 0",color:GRAY3,fontSize:13,fontStyle:"italic"}}>How can I help you?</div>
              </div>
            ):(
              <div style={{maxWidth:780,margin:"0 auto"}}>
                {msgs.map((m,i)=>(
                  <Msg key={i} msg={m}
                    streaming={isWorking&&i===msgs.length-1&&m.role==="assistant"&&!m.frameworkData&&!m.treeData}
                    onRegenerate={m.role==="assistant"&&i===msgs.length-1?()=>{
                      const lastUser=msgs.slice(0,i).reverse().find(x=>x.role==="user");
                      if(lastUser) send(lastUser.content);
                    }:null}
                    onEdit={m.role==="assistant"?instruction=>{
                      const lastUser=msgs.slice(0,i).reverse().find(x=>x.role==="user");
                      if(lastUser) send(lastUser.content+" [CHANGE REQUEST: "+instruction+"] Please rewrite your previous response with this specific change applied.");
                    }:null}
                  />
                ))}
                {(fwLoading||treeLoading)&&(
                  <div style={{display:"flex",gap:9,marginBottom:12,alignItems:"center"}}>
                    <div style={{width:24,height:24,border:"1.5px solid "+G_DARK,background:G_DARK,display:"flex",alignItems:"center",justifyContent:"center",fontSize:8,fontWeight:700,color:WHITE}}>CC</div>
                    <div style={{background:WHITE,border:"1.5px solid "+G_BOR,padding:"9px 13px",fontSize:12,color:GRAY2,display:"flex",alignItems:"center",gap:7}}>
                      <div style={{width:10,height:10,border:"2px solid "+G_LIGHT,borderTopColor:G_DARK,borderRadius:"50%",animation:"spin 0.8s linear infinite",flexShrink:0}}/>
                      {fwLoading?"Building MECE framework diagram...":"Building market sizing tree..."}
                    </div>
                  </div>
                )}
                <div ref={endRef}/>
              </div>
            )}
          </div>

          {/* INPUT */}
          <div style={{padding:"10px 16px 13px",background:WHITE,borderTop:"1.5px solid "+G_BOR,flexShrink:0}}>
            <div style={{display:"flex",gap:7,alignItems:"flex-end",background:G_PALE,border:"1.5px solid "+(isWorking?"#2980b9":input.length>0?G_DARK:G_BOR),padding:"8px 9px 8px 12px",transition:"border-color 0.15s"}}>
              <button onClick={()=>fileRef.current?.click()} title="Attach document" style={{background:"none",border:"none",color:G_MID,cursor:"pointer",fontSize:14,padding:"1px 3px",flexShrink:0}} onMouseEnter={e=>e.target.style.color=G_DARK} onMouseLeave={e=>e.target.style.color=G_MID}>+</button>
              <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.doc,.docx" style={{display:"none"}} onChange={handleAttach}/>
              <textarea ref={taRef} value={input} onChange={e=>setInput(e.target.value)} onKeyDown={handleKey} disabled={isWorking}
                placeholder={activeMod==="case"?"Describe your client problem — MECE framework auto-generates...":activeMod==="sizing"?"Enter any market sizing or guestimate problem...":activeMod==="framework"?"Describe the problem — MECE diagram auto-generates...":activeMod==="deck"?"Type anything — builds CEO report from your session work...":"Type your input... + to attach a document (Enter to send)"}
                rows={1} style={{flex:1,background:"transparent",border:"none",color:BLACK,fontSize:13,lineHeight:1.6,resize:"none",minHeight:22,maxHeight:120,padding:0,caretColor:G_DARK,fontFamily:"Arial,sans-serif"}}/>
              <button onClick={()=>send(input)} disabled={isWorking||!input.trim()} style={{padding:"5px 13px",flexShrink:0,background:isWorking||!input.trim()?GRAY4:G_DARK,border:"none",color:WHITE,fontFamily:"Arial,sans-serif",fontSize:11,fontWeight:700,cursor:isWorking||!input.trim()?"default":"pointer",transition:"all 0.12s"}}>
                {isWorking?"...":"Send"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {showMemory&&<MemoryModal cases={savedCases} onSelect={c=>{setActiveMod(c.module);setSessions(p=>({...p,[c.module]:c.history}));setShowMemory(false);}} onDelete={async id=>{await dropCase(id);setSavedCases(p=>p.filter(c=>c.id!==id));}} onClose={()=>setShowMemory(false)}/>}
      {showKB&&<KBModal kb={kb} kbFiles={kbFiles} setKbFiles={setKbFiles} onClose={()=>setShowKB(false)}/>}
    </div>
  );
}
