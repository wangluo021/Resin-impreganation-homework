# MECHENG 451 Fiber Impregnation Homework

This repository contains the student-facing version of the 2D fiber-scale resin impregnation homework package.

## Contents

- `student_template/`: Python homework template for students.
- `docs/MECHENG451_fiber_impregnation_homework.docx`: assignment handout with UMich table colors.
- `docs/MECHENG451_result_examples.docx`: example result figures handout.
- `homework_questions.md`: written-response questions.
- `requirements.txt`: Python dependencies.

The instructor solution, answer key, grading tolerances, and full expected-results directory are intentionally not included in this student-facing repository.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python student_template\main.py
```

The template stops with a clear `NotImplementedError` until the required TODO functions are completed.

Students complete the geometry, viscosity, and impregnation functions listed in the assignment handout. The cure kinetics and cure-viscosity coupling are supplied so the homework stays focused on fiber architecture, processing temperature, resin mobility, and impregnation during manufacturing.
