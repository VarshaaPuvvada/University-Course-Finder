import React, { useEffect, useState } from "react";
import { BarChart3, FileText, Image, Loader2, Mic, Search } from "lucide-react";
import { analytics, recommend, upload } from "./api/client";
import "./styles/app.css";

const initialForm = {
  query: "I want to learn how AI can predict stock prices",
  currentSkills: "Python Programming",
  studentLevel: "beginner",
  careerGoal: "Machine learning engineer in finance",
  topK: 5,
};

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    analytics().then(setStats).catch(() => {});
  }, []);

  async function submitSearch(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const data = await recommend({
        query: form.query,
        current_skills: splitSkills(form.currentSkills),
        student_level: form.studentLevel,
        career_goal: form.careerGoal || null,
          top_k: Number(form.topK),
          organizations: splitSkills(form.organizations || ""),
          difficulties: splitSkills(form.difficulties || ""),
          skill_categories: splitSkills(form.skillCategories || ""),
          min_rating: form.minRating ? Number(form.minRating) : null,
          strict_difficulty: Boolean(form.strictDifficulty),
          use_llm_judge: Boolean(form.useLlmJudge),
      });
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitFile(endpoint, file) {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const data = await upload(endpoint, file, form);
      setResult(data.recommendation);
      if (data.transcript) setForm((current) => ({ ...current, query: data.transcript }));
      if (data.extracted_text) setForm((current) => ({ ...current, query: data.extracted_text }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <form className="query-panel" onSubmit={submitSearch}>
          <div className="panel-header">
            <h1>University Course Finder</h1>
            <button className="primary-button" disabled={loading}>
              {loading ? <Loader2 className="spin" size={18} /> : <Search size={18} />}
              Search
            </button>
          </div>

          <textarea
            value={form.query}
            onChange={(event) => setForm({ ...form, query: event.target.value })}
            aria-label="Learning goal"
          />

          <div className="controls">
            <label>
              Current skills
              <input
                value={form.currentSkills}
                onChange={(event) => setForm({ ...form, currentSkills: event.target.value })}
              />
            </label>
            <label>
              Student level
              <select
                value={form.studentLevel}
                onChange={(event) => setForm({ ...form, studentLevel: event.target.value })}
              >
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </label>
            <label>
              Career goal
              <input
                value={form.careerGoal}
                onChange={(event) => setForm({ ...form, careerGoal: event.target.value })}
              />
            </label>
            <label>
              Results
              <input
                type="number"
                min="1"
                max="20"
                value={form.topK}
                onChange={(event) => setForm({ ...form, topK: event.target.value })}
              />
            </label>
            <label>
              Min rating
              <input
                type="number"
                min="0"
                max="5"
                step="0.1"
                value={form.minRating || ""}
                onChange={(event) => setForm({ ...form, minRating: event.target.value })}
              />
            </label>
            <label>
              Organizations
              <input
                value={form.organizations || ""}
                onChange={(event) => setForm({ ...form, organizations: event.target.value })}
              />
            </label>
            <label>
              Skill filter
              <input
                value={form.skillCategories || ""}
                onChange={(event) => setForm({ ...form, skillCategories: event.target.value })}
              />
            </label>
            <label>
              Difficulty filter
              <input
                value={form.difficulties || ""}
                onChange={(event) => setForm({ ...form, difficulties: event.target.value })}
              />
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={Boolean(form.strictDifficulty)}
                onChange={(event) => setForm({ ...form, strictDifficulty: event.target.checked })}
              />
              Strict level
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={Boolean(form.useLlmJudge)}
                onChange={(event) => setForm({ ...form, useLlmJudge: event.target.checked })}
              />
              LLM judge
            </label>
          </div>

          <div className="upload-row">
            <UploadButton icon={<Mic size={18} />} label="Audio" accept="audio/*" onFile={(file) => submitFile("/speech-query", file)} />
            <UploadButton icon={<FileText size={18} />} label="PDF" accept="application/pdf" onFile={(file) => submitFile("/upload-pdf", file)} />
            <UploadButton icon={<Image size={18} />} label="Image" accept="image/*" onFile={(file) => submitFile("/upload-image", file)} />
          </div>
        </form>

        {error && <p className="error">{error}</p>}

        <section className="results-grid">
          <div className="recommendations">
            <h2>Recommendations</h2>
            {result?.advisor_summary && <p className="advisor-summary">{result.advisor_summary}</p>}
            {result?.validation_warnings?.length > 0 && (
              <div className="warning-list">
                {result.validation_warnings.map((warning) => <span key={warning}>{warning}</span>)}
              </div>
            )}
            {result?.recommendations?.map((course) => (
              <article className="course-card" key={`${course.title}-${course.organization}`}>
                <div>
                  <h3>{course.title}</h3>
                  <dl className="field-list compact">
                    <div><dt>Organization</dt><dd>{course.organization}</dd></div>
                    <div><dt>Difficulty</dt><dd>{course.difficulty}</dd></div>
                    <div><dt>Rating</dt><dd>{course.rating ?? "N/A"}</dd></div>
                    <div><dt>Duration</dt><dd>{course.duration || "N/A"}</dd></div>
                  </dl>
                </div>
                <div className="field-block">
                  <span>Why this course</span>
                  <p>{course.explanation}</p>
                </div>
                {course.description && (
                  <div className="field-block">
                    <span>Course description</span>
                    <p>{course.description}</p>
                  </div>
                )}
                <div className="course-meta">
                  <span>Type: {course.course_type || "Course"}</span>
                  <span>Reviews: {Math.round(course.review_count || 0).toLocaleString()}</span>
                  <span>Score: {course.final_score}</span>
                  {course.llm_enhanced && <span>LLM enhanced</span>}
                </div>
                <div className="field-block">
                  <span>Skills</span>
                  <div className="skill-list">
                    {course.skills.slice(0, 6).map((skill) => <span key={skill}>{skill}</span>)}
                  </div>
                </div>
                {course.prerequisite_gaps.length > 0 && (
                  <p className="gap">Prerequisite gaps: {course.prerequisite_gaps.join(", ")}</p>
                )}
                {course.url && <a href={course.url} target="_blank" rel="noreferrer">Open course</a>}
              </article>
            ))}
          </div>

          <aside className="side-panel">
            <h2>Learning Path</h2>
            <ol className="path-list">
              {(result?.learning_path || []).map((step) => <li key={step}>{step}</li>)}
            </ol>
            <h2>Analytics</h2>
            <div className="analytics-block">
              <BarChart3 size={18} />
              <span>{stats?.total_courses || 0} courses indexed</span>
            </div>
            <div className="mini-list">
              {(stats?.popular_skills || []).slice(0, 6).map((item) => (
                <span key={item.skill}>{item.skill}: {item.count}</span>
              ))}
            </div>
          </aside>
        </section>
      </section>
    </main>
  );
}

function splitSkills(value) {
  return value.split(",").map((skill) => skill.trim()).filter(Boolean);
}

function UploadButton({ icon, label, accept, onFile }) {
  return (
    <label className="upload-button">
      {icon}
      {label}
      <input
        type="file"
        accept={accept}
        onChange={(event) => onFile(event.target.files?.[0])}
      />
    </label>
  );
}
