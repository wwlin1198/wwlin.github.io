---
layout: publication
title: "Asynchronous Value Factorization for Multi-Robot Cooperative Reinforcement Learning"
authors: "Enrico Marchesini, Wo Wei Lin, Priya Donti, Christopher Amato"
venue: "IROS"
year: 2026
paper_url: "TBD"
type: "conference"
---
Value factorization methods such as VDN, QMIX, and QPLEX, are widely used for cooperative multi-robot reinforcement learning. However, existing approaches assume synchronous, single-step (primitive) action selection, which is incompatible with real robotic systems where high-level behaviors are executed as temporally extended macro-actions of variable duration. This asynchronous execution introduces temporal misalignment between action selection and value conditioning, breaking the assumptions underlying value factorization. We introduce Asynchronous Value Factorization (AVF), a framework that extends value factorization to macro-action settings by (i) conditioning value updates on macro-action termination events and (ii) introducing a macro-state buffer that preserves temporal consistency between decision context and centralized training signals. AVF maintains decentralized execution while enabling correct credit assignment under asynchronous decision timing.
We instantiate AVF with VDN, QMIX, and QPLEX mixers and evaluate it on standard multi-robot macro-action benchmarks and a real-world box-pushing task with Turtlebot platforms. Across tasks with increasing coordination complexity, AVF significantly outperforms primitive-action baselines and prior macro-action methods.
