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
  body.append("file", file);
  body.append("current_skills", form.currentSkills);
  body.append("student_level", form.studentLevel);
  body.append("career_goal", form.careerGoal);
  body.append("top_k", String(form.topK));
  body.append("organizations", form.organizations || "");
  body.append("difficulties", form.difficulties || "");
  body.append("skill_categories", form.skillCategories || "");
  body.append("min_rating", form.minRating || "");
  body.append("strict_difficulty", String(Boolean(form.strictDifficulty)));
  body.append("use_llm_judge", String(Boolean(form.useLlmJudge)));
  const response = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
