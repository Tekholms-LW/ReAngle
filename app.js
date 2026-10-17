const M='media/';
const CLIPS={
 sweater:{name:'Woman in sweater',orig:'sweater_original',note:'First 1.5 s of the original, re-shot five ways. 448×448 renders, shown upscaled.',
  moves:[['swing_right','Swing right'],['pull_back','Pull-back'],['orbit_right','Orbit right'],['orbit_left','Orbit left'],['arc_in','Arc in']]},
 fog:{name:'Woman in fog',orig:'fog_original',note:'4 s clip, 12° orbit rendered in overlapping chunks. Toggle the fill to see the ghost it removes.',
  moves:[['orbit_left','Orbit left 12°']],fill:true}};
const $=id=>document.getElementById(id),A=$('vA'),B=$('vB'),stage=$('stage');
let clip='sweater',move=0;
function btns(el,items,cur,fn){el.innerHTML='';items.forEach(([k,l],i)=>{const b=document.createElement('button');b.textContent=l;if(i===cur)b.className='on';b.onclick=()=>fn(i);el.appendChild(b)})}
function load(){
 const c=CLIPS[clip];const keys=Object.keys(CLIPS);
 btns($('clips'),keys.map(k=>[k,CLIPS[k].name]),keys.indexOf(clip),i=>{clip=keys[i];move=0;load()});
 btns($('moves'),c.moves,move,i=>{move=i;load()});
 $('fillWrap').hidden=!c.fill;
 const mk=c.moves[move][0];const suf=(c.fill&&!$('fill').checked)?'_nofill':'';
 A.poster=M+c.orig+'.jpg';A.src=M+c.orig+'.mp4';B.src=M+clip+'_'+mk+suf+'.mp4';
 $('lblR').textContent='Re-shot · '+c.moves[move][1]+(suf?' (no fill)':'');$('note').textContent=c.note;
 Promise.all([A,B].map(v=>new Promise(r=>v.addEventListener('canplay',r,{once:true})))).then(()=>{A.currentTime=B.currentTime=0;if($('play').textContent==='Pause'){A.play();B.play()}});
}
// keep B synced to A
setInterval(()=>{if(!A.paused&&Math.abs(B.currentTime-A.currentTime)>0.06)B.currentTime=A.currentTime},250);
A.addEventListener('play',()=>B.play());A.addEventListener('pause',()=>B.pause());
A.addEventListener('ended',()=>B.pause());
$('play').onclick=()=>{if(A.paused){A.play();$('play').textContent='Pause'}else{A.pause();$('play').textContent='Play'}};
$('loop').onchange=e=>{A.loop=B.loop=e.target.checked;if(e.target.checked&&A.ended){A.play()}};
$('sbs').onchange=e=>stage.classList.toggle('sbs',e.target.checked);
$('fill').onchange=load;
function setX(x){const r=stage.getBoundingClientRect();const p=Math.min(100,Math.max(0,(x-r.left)/r.width*100));$('topWrap').style.clipPath=`inset(0 0 0 ${p}%)`;$('divider').style.left=p+'%'}
let drag=false;stage.addEventListener('pointerdown',e=>{if(stage.classList.contains('sbs'))return;drag=true;stage.setPointerCapture(e.pointerId);setX(e.clientX)});
stage.addEventListener('pointermove',e=>{if(drag)setX(e.clientX)});stage.addEventListener('pointerup',()=>drag=false);
$('sbs').addEventListener('change',e=>{$('topWrap').style.clipPath=e.target.checked?'none':'inset(0 0 0 50%)';$('divider').style.left='50%'});
load();
