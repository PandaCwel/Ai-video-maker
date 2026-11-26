const $ = (selector) => document.querySelector(selector);
const projectsEl = $("#projects");
const dialog = $("#project-dialog");

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
}

async function request(url, options = {}) {
  const response = await fetch(url, {headers: {"Content-Type":"application/json"}, ...options});
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

async function loadProjects() {
  try {
    const {projects} = await request("/api/projects");
    $("#project-count").textContent = `${projects.length} project${projects.length === 1 ? "" : "s"}`;
    if (!projects.length) {
      projectsEl.innerHTML = '<div class="empty-state">No projects yet. Create your first film project to get started.</div>';
      return;
    }
    projectsEl.innerHTML = projects.map(project => `
      <article class="project-card" tabindex="0" data-id="${escapeHtml(project.id)}">
        <div class="project-icon">✦</div>
        <h3>${escapeHtml(project.title)}</h3>
        <p>${escapeHtml(project.status)} · Updated ${new Date(project.updated_at).toLocaleDateString()}</p>
      </article>`).join("");
    projectsEl.querySelectorAll(".project-card").forEach(card => {
      card.addEventListener("click", () => openProject(card.dataset.id));
      card.addEventListener("keydown", event => { if (event.key === "Enter") openProject(card.dataset.id); });
    });
  } catch (error) {
    projectsEl.innerHTML = `<div class="empty-state">Could not load projects: ${escapeHtml(error.message)}</div>`;
  }
}

async function openProject(id) {
  try {
    const project = await request(`/api/projects/${encodeURIComponent(id)}`);
    const script = prompt(`Project: ${project.title}\n\nEdit screenplay text (saved when you press OK):`, project.script || "");
    if (script !== null) {
      await request(`/api/projects/${encodeURIComponent(id)}/script`, {method:"PUT", body:JSON.stringify({script})});
      alert("Screenplay saved.");
      await loadProjects();
    }
  } catch (error) { alert(error.message); }
}

$("#new-project").addEventListener("click", () => { $("#project-title").value = ""; dialog.showModal(); });
$("#project-form").addEventListener("submit", async event => {
  event.preventDefault();
  const title = $("#project-title").value.trim();
  if (!title) return;
  try {
    await request("/api/projects", {method:"POST", body:JSON.stringify({title})});
    dialog.close();
    await loadProjects();
  } catch (error) { alert(error.message); }
});
loadProjects();
