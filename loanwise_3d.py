import streamlit as st
import streamlit.components.v1 as components


def render_3d_hero(customer_count: int = 5000):
    """Render a lightweight 3D fintech hero without changing the ML pipeline."""
    components.html(
        r"""
        <!doctype html>
        <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <style>
            *{box-sizing:border-box}
            html,body{margin:0;background:transparent;font-family:Inter,Arial,sans-serif;color:#f8fbff}
            .hero{position:relative;height:330px;overflow:hidden;border-radius:26px;
              border:1px solid rgba(115,184,255,.25);
              background:radial-gradient(circle at 78% 48%,rgba(14,165,233,.20),transparent 30%),
              radial-gradient(circle at 10% 0%,rgba(59,130,246,.22),transparent 35%),
              linear-gradient(120deg,#071426 0%,#0b2340 55%,#071827 100%);
              box-shadow:0 24px 70px rgba(0,0,0,.28)}
            #scene{position:absolute;inset:0;width:100%;height:100%;opacity:.95}
            .copy{position:relative;z-index:2;padding:38px 42px;max-width:650px;pointer-events:none}
            .eyebrow{display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:30px;
              color:#9edcff;background:rgba(56,189,248,.10);border:1px solid rgba(125,211,252,.25);
              font-size:10px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase}
            .dot{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 12px #34d399}
            h1{font-size:clamp(32px,5vw,54px);line-height:1.02;letter-spacing:-2px;margin:20px 0 12px;font-weight:850}
            h1 span{background:linear-gradient(90deg,#60a5fa,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
            p{max-width:490px;color:#bfd2e8;font-size:13px;line-height:1.8;margin:0}
            .chips{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px}
            .chip{font-size:10px;color:#d9eaff;padding:8px 11px;border-radius:10px;
              background:rgba(15,35,60,.72);border:1px solid rgba(148,190,230,.17)}
            .orbital-label{position:absolute;right:7%;bottom:25px;z-index:2;color:#9bc9ef;font-size:10px;
              letter-spacing:1.4px;text-transform:uppercase;opacity:.85}
            @media(max-width:650px){.hero{height:310px}.copy{padding:28px 22px}.orbital-label{display:none}#scene{opacity:.55}}
            @media(prefers-reduced-motion:reduce){*{animation:none!important}}
          </style>
        </head>
        <body>
          <section class="hero">
            <canvas id="scene"></canvas>
            <div class="copy">
              <div class="eyebrow"><span class="dot"></span> Intelligent banking insights</div>
              <h1>See the signal.<br><span>Know your customer.</span></h1>
              <p>Explainable machine learning for personal-loan campaign insights. Explore customer patterns, estimate offer acceptance likelihood, and understand what drives each prediction.</p>
              <div class="chips">
                <span class="chip">✦ Predictive intelligence</span>
                <span class="chip">◈ Explainable AI</span>
                <span class="chip">⌁ Customer analytics</span>
              </div>
            </div>
            <div class="orbital-label">LOANWISE AI · INTELLIGENCE ENGINE</div>
          </section>
          <script type="module">
            import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
            const canvas=document.getElementById('scene');
            const scene=new THREE.Scene();
            const camera=new THREE.PerspectiveCamera(42,1,.1,100);
            camera.position.set(0,0,8);
            const renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:true});
            renderer.setPixelRatio(Math.min(devicePixelRatio,2));
            const group=new THREE.Group(); group.position.set(2.55,.05,0); scene.add(group);
            const core=new THREE.Mesh(new THREE.IcosahedronGeometry(1.22,3),
              new THREE.MeshStandardMaterial({color:0x1687d4,metalness:.72,roughness:.22,
                emissive:0x063b70,emissiveIntensity:.65,wireframe:false}));
            group.add(core);
            const wire=new THREE.Mesh(new THREE.IcosahedronGeometry(1.27,2),
              new THREE.MeshBasicMaterial({color:0x69d7ff,wireframe:true,transparent:true,opacity:.24}));
            group.add(wire);
            const ringMat=new THREE.MeshBasicMaterial({color:0x46c8ff,transparent:true,opacity:.68});
            for(let i=0;i<3;i++){
              const ring=new THREE.Mesh(new THREE.TorusGeometry(1.72+i*.16,.008,8,180),ringMat);
              ring.rotation.set(.8+i*.52,.35+i*.7,.3); group.add(ring);
            }
            const starGeo=new THREE.BufferGeometry(), count=850, positions=new Float32Array(count*3);
            for(let i=0;i<count;i++){positions[i*3]=(Math.random()-.5)*12;positions[i*3+1]=(Math.random()-.5)*6;positions[i*3+2]=(Math.random()-.5)*5}
            starGeo.setAttribute('position',new THREE.BufferAttribute(positions,3));
            const stars=new THREE.Points(starGeo,new THREE.PointsMaterial({color:0x8acfff,size:.018,transparent:true,opacity:.7}));
            scene.add(stars);
            scene.add(new THREE.AmbientLight(0x9bcfff,1.1));
            const light=new THREE.PointLight(0x35bfff,30,15); light.position.set(3,3,4); scene.add(light);
            function resize(){const w=canvas.clientWidth,h=canvas.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
            new ResizeObserver(resize).observe(canvas); resize();
            let pointer=0;
            window.addEventListener('pointermove',e=>{pointer=(e.clientX/window.innerWidth-.5)*.18});
            function animate(t){requestAnimationFrame(animate);const s=t*.00035;group.rotation.y=s+pointer;group.rotation.x=Math.sin(s*.7)*.12;wire.rotation.y=-s*.7;stars.rotation.y=s*.05;renderer.render(scene,camera)}
            if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches) animate(0); else renderer.render(scene,camera);
          </script>
        </body>
        </html>
        """,
        height=345,
        scrolling=False,
    )
