const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function recommend(payload) {
  const response = await fetch(`${API_BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function analytics() {
  const response = await fetch(`${API_BASE}/analytics`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function upload(endpoint, file, form) {
  const body = new FormData();
  const topK = form.topK ? Number(form.topK) : 5;
  body.append("file", file);
  body.append("current_skills", form.currentSkills);
  body.append("student_level", form.studentLevel || "beginner");
  body.append("career_goal", form.careerGoal);
  body.append("top_k", String(Math.min(Math.max(topK, 1), 20)));
  body.append("topK", String(Math.min(Math.max(topK, 1), 20)));
  body.append("organizations", "");
  body.append("difficulties", "");
  body.append("skill_categories", "");
  body.append("min_rating", "");
  body.append("strict_difficulty", "false");
  body.append("use_llm_judge", "false");
  body.append("preferred_skills", "");
  body.append("completed_courses", "");
  body.append("liked_courses", "");
  body.append("disliked_courses", "");
  body.append("learner_progress", "");
  body.append("peer_group", "");
  const response = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
