"use client";

import { useMemo, useState } from "react";

type Result = {
  ssim: number;
  ssim_verdict: string;
  cnn_match_prob: number;
  cnn_verdict: string;
};

const API_URL = "http://localhost:8000/compare";

function Verdict({ text }: { text: string }) {
  const ok = text === "High Match";
  return (
    <span
      className={`inline-block rounded-full px-3 py-1 text-sm font-semibold ${
        ok ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
      }`}
    >
      {ok ? "High Match (Production Ready)" : "Visual Mismatch (Needs Fix)"}
    </span>
  );
}

function Upload({
  label,
  file,
  onChange,
}: {
  label: string;
  file: File | null;
  onChange: (f: File | null) => void;
}) {
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 font-semibold text-gray-900">{label}</h2>
      <div className="mb-3 flex h-40 items-center justify-center overflow-hidden rounded-lg bg-gray-100">
        {url ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt={label} className="max-h-full max-w-full object-contain" />
        ) : (
          <span className="text-sm text-gray-500">No image selected</span>
        )}
      </div>
      <input
        type="file"
        accept="image/png,image/jpeg"
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
        className="block w-full text-sm text-gray-700 file:mr-3 file:rounded-md file:border-0 file:bg-gray-900 file:px-3 file:py-2 file:text-sm file:text-white hover:file:bg-gray-700"
      />
    </div>
  );
}

export default function Home() {
  const [design, setDesign] = useState<File | null>(null);
  const [code, setCode] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function compare() {
    if (!design || !code) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("design", design);
      form.append("code", code);
      const res = await fetch(API_URL, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server error ${res.status}`);
      setResult(await res.json());
    } catch {
      setError("Could not get a result. Is the Python API running on port 8000?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 px-6 py-10 text-gray-900">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold">UI-Vision</h1>
        <p className="mt-1 mb-8 text-gray-600">
          Compare a Figma design with a screenshot of the coded page.
        </p>

        <div className="grid gap-4 md:grid-cols-2">
          <Upload label="Figma design" file={design} onChange={setDesign} />
          <Upload label="Coded page screenshot" file={code} onChange={setCode} />
        </div>

        <button
          onClick={compare}
          disabled={!design || !code || loading}
          className="mt-6 w-full rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          {loading ? "Comparing..." : "Compare"}
        </button>

        {error && (
          <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>
        )}

        {result && (
          <div className="mt-8 grid gap-4 md:grid-cols-2">
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <p className="text-sm text-gray-500">Baseline (SSIM)</p>
              <p className="my-2 text-3xl font-bold">{result.ssim.toFixed(3)}</p>
              <Verdict text={result.ssim_verdict} />
              <p className="mt-3 text-xs text-gray-500">Similarity score, 1.0 means identical.</p>
            </div>
            <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
              <p className="text-sm text-gray-500">CNN (ResNet18)</p>
              <p className="my-2 text-3xl font-bold">
                {(result.cnn_match_prob * 100).toFixed(1)}%
              </p>
              <Verdict text={result.cnn_verdict} />
              <p className="mt-3 text-xs text-gray-500">Probability that the design and code match.</p>
            </div>
          </div>
        )}

        <p className="mt-10 text-xs text-gray-500">
          Note: the model was trained on component-sized images (up to 704×256 px).
          Full-page screenshots are not supported yet.
        </p>
      </div>
    </main>
  );
}