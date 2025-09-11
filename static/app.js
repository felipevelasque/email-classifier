const pick = document.getElementById('pick');
const fileInput = document.getElementById('file');
const dropzone = document.getElementById('dropzone');
const analyzeBtn = document.getElementById('analyze');
const textArea = document.getElementById('text');
const result = document.getElementById('result');
const badge = document.getElementById('badge');
const conf = document.getElementById('confidence');
const chips = document.getElementById('signals');
const reply = document.getElementById('reply');
const copyBtn = document.getElementById('copy');
const copyClearBtn = document.getElementById('copyClear');
const toast = document.getElementById('toast');
const meta = document.getElementById('meta');

let droppedFile = null;


function showToast(msg) {
    toast.textContent = msg;
    toast.classList.remove('hidden');
    setTimeout(() => toast.classList.add('hidden'), 1800);
}

pick.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    droppedFile = e.target.files[0] || null;
    if (droppedFile) showToast(`Arquivo selecionado: ${droppedFile.name}`);
});

;['dragenter','dragover'].forEach(ev => dropzone.addEventListener(ev, (e)=>{
    e.preventDefault(); e.stopPropagation(); dropzone.classList.add('drag');
}));
;['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, (e)=>{
    e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('drag');
}));

dropzone.addEventListener('drop', (e) => {
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    if (!/\.(txt|pdf)$/i.test(f.name)) { showToast('Envie .txt ou .pdf'); return; }
    droppedFile = f;
    showToast(`Arquivo solto: ${f.name}`);
});

async function analyze() {
    analyzeBtn.disabled = true; analyzeBtn.textContent = 'Processando…';
    try {
    const fd = new FormData();
    if (droppedFile) {
        fd.append('email_file', droppedFile);
    } else {
        const txt = (textArea.value || '').trim();
        if (!txt) { showToast('Cole um texto ou envie um arquivo.'); return; }
        fd.append('email_text', txt);
    }

    const res = await fetch('/api/analyze', { method: 'POST', body: fd });
    if (!res.ok) {
        const err = await res.json().catch(()=>({detail: 'Erro desconhecido'}));
        throw new Error(err.detail || 'Falha ao processar');
    }
    const data = await res.json();
    renderResult(data);
    } catch (e) {
        showToast(e.message);
    } finally {
        analyzeBtn.disabled = false; analyzeBtn.textContent = 'Analisar';
    }
}

function renderResult(data) {
    result.classList.remove('hidden');
    badge.textContent = data.category || '—';
    badge.classList.remove('success','neutral');
    badge.classList.add(data.category === 'Produtivo' ? 'success' : 'neutral');
    conf.textContent = `Confiança: ${(data.confidence*100).toFixed(0)}%`;
    chips.innerHTML = '';
    (data.meta?.signals || []).forEach(s => {
        const el = document.createElement('span');
        el.className = 'chip'; el.textContent = s; chips.appendChild(el);
    });
    reply.value = data.reply || '';
    meta.textContent = `Usou HF: ${data.meta?.used_hf ? 'sim' : 'não'} | Usou LLM: ${data.meta?.used_openai ? 'sim' : 'não'} | Fallbacks: ${(data.meta?.fallbacks||[]).join(', ')}`;
}


analyzeBtn.addEventListener('click', analyze);
copyBtn.addEventListener('click', async () => {
    await navigator.clipboard.writeText(reply.value || '');
    showToast('Resposta copiada!');
});
copyClearBtn.addEventListener('click', async () => {
    await navigator.clipboard.writeText(reply.value || '');
    textArea.value = ''; droppedFile = null; document.getElementById('file').value = '';
    showToast('Copiado e limpo!');
});