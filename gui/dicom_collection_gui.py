# -*- coding: utf-8 -*-
"""Tkinter GUI for DICOM collection queue (gangnam 1xxxx studies)."""
from __future__ import annotations

import csv
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

_V1_ROOT = Path(__file__).resolve().parents[1]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.data.collection_manifest import build_manifest, write_manifest

DEFAULT_WORK = Path(r"<REDACTED_PATH> ViT_Infectivity")
DEFAULT_RAW = Path(r"<REDACTED_PATH> CXR_Active Image")
DEFAULT_MANIFEST = _V1_ROOT / "artifacts" / "dicom_collection_manifest.csv"
STATUS_LABELS = {
    "ready": "ready (work dir)",
    "pending_export": "RAW only -> export needed",
    "pending_collect": "no DICOM",
}


class DicomCollectionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Infectivity ViT - DICOM Collection")
        self.geometry("980x640")
        self.manifest_path = DEFAULT_MANIFEST
        self.rows: list[dict[str, str]] = []
        self.filtered: list[dict[str, str]] = []
        self._build_ui()
        self.refresh_manifest()

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Refresh manifest", command=self.refresh_manifest).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Export pending CSV", command=self.export_pending).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Run prepare_dicom", command=self.run_prepare).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open work dir", command=lambda: self._open_dir(DEFAULT_WORK)).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Open RAW dir", command=lambda: self._open_dir(DEFAULT_RAW)).pack(side=tk.LEFT, padx=4)

        filt = ttk.Frame(self, padding=(8, 0))
        filt.pack(fill=tk.X)
        ttk.Label(filt, text="Filter:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="pending_collect")
        for val, label in [
            ("all", "all"),
            ("pending_collect", "pending_collect"),
            ("pending_export", "pending_export"),
            ("ready", "ready"),
        ]:
            ttk.Radiobutton(filt, text=label, variable=self.status_var, value=val, command=self.apply_filter).pack(
                side=tk.LEFT, padx=6,
            )

        self.summary_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.summary_var, padding=8).pack(fill=tk.X)

        cols = ("study_no", "status", "n_slices_work", "n_slices_raw", "has_sputum", "has_ct", "n_labeled_heads", "note")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=22)
        headings = {
            "study_no": "Study",
            "status": "Status",
            "n_slices_work": "Work slices",
            "n_slices_raw": "RAW slices",
            "has_sputum": "Sputum",
            "has_ct": "CT",
            "n_labeled_heads": "# heads",
            "note": "Note",
        }
        widths = {"study_no": 70, "status": 140, "n_slices_work": 90, "n_slices_raw": 90,
                  "has_sputum": 60, "has_ct": 40, "n_labeled_heads": 60, "note": 220}
        for c in cols:
            self.tree.heading(c, text=headings[c])
            self.tree.column(c, width=widths[c], anchor=tk.W)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.tree.bind("<Double-1>", self._on_double_click)

        note_frame = ttk.Frame(self, padding=8)
        note_frame.pack(fill=tk.X)
        ttk.Label(note_frame, text="Note for selected study:").pack(side=tk.LEFT)
        self.note_entry = ttk.Entry(note_frame)
        self.note_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(note_frame, text="Save note", command=self.save_note).pack(side=tk.LEFT)

    def refresh_manifest(self) -> None:
        labels_csv = _V1_ROOT / "artifacts" / "labels_v1.csv"
        if not labels_csv.exists():
            messagebox.showerror("Missing labels", f"Not found: {labels_csv}\nRun run_build_labels.ps1 first.")
            return
        self.rows = build_manifest(
            labels_csv=labels_csv,
            work_dir=DEFAULT_WORK,
            raw_dir=DEFAULT_RAW,
            sputum_dir=Path(r"<REDACTED_PATH> SPUTUM DATA"),
            ct_dir=Path(r"<REDACTED_PATH> CT Reading Collection GUI"),
        )
        write_manifest(self.rows, self.manifest_path)
        ready = sum(1 for r in self.rows if r["status"] == "ready")
        pending_export = sum(1 for r in self.rows if r["status"] == "pending_export")
        pending_collect = sum(1 for r in self.rows if r["status"] == "pending_collect")
        self.summary_var.set(
            f"Total {len(self.rows)} | ready {ready} | pending_export {pending_export} | pending_collect {pending_collect}"
        )
        self.apply_filter()

    def apply_filter(self) -> None:
        filt = self.status_var.get()
        if filt == "all":
            self.filtered = list(self.rows)
        else:
            self.filtered = [r for r in self.rows if r["status"] == filt]
        for item in self.tree.get_children():
            self.tree.delete(item)
        for r in self.filtered:
            self.tree.insert("", tk.END, values=tuple(r[c] for c in (
                "study_no", "status", "n_slices_work", "n_slices_raw",
                "has_sputum", "has_ct", "n_labeled_heads", "note",
            )))

    def _selected_row(self) -> dict[str, str] | None:
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        study = vals[0]
        for r in self.rows:
            if r["study_no"] == study:
                return r
        return None

    def _on_double_click(self, _event: tk.Event) -> None:
        row = self._selected_row()
        if not row:
            return
        self.clipboard_clear()
        self.clipboard_append(row["study_no"])
        self.note_entry.delete(0, tk.END)
        self.note_entry.insert(0, row.get("note", ""))

    def save_note(self) -> None:
        row = self._selected_row()
        if not row:
            messagebox.showinfo("Select study", "Select a study in the table first.")
            return
        row["note"] = self.note_entry.get().strip()
        write_manifest(self.rows, self.manifest_path)
        self.apply_filter()

    def export_pending(self) -> None:
        pending = [r for r in self.rows if r["status"] != "ready"]
        out = _V1_ROOT / "artifacts" / "dicom_collection_pending.csv"
        write_manifest(pending, out)
        messagebox.showinfo("Exported", f"{len(pending)} studies ->\n{out}")

    def run_prepare(self) -> None:
        script = _V1_ROOT / "scripts" / "prepare_dicom_gangnam.ps1"
        if not script.exists():
            messagebox.showerror("Missing script", str(script))
            return
        subprocess.Popen(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            cwd=str(_V1_ROOT),
        )
        messagebox.showinfo("Started", "prepare_dicom_gangnam.ps1 launched in background.\nClick Refresh after it finishes.")

    @staticmethod
    def _open_dir(path: Path) -> None:
        if not path.exists():
            messagebox.showerror("Not found", str(path))
            return
        subprocess.Popen(["explorer", str(path)])


def main() -> None:
    app = DicomCollectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
