---
layout: publication
title: "Asynchronous Value Factorization for Multi-Robot Cooperative Reinforcement Learning"
authors: "Wo Wei Lin, Ethan Rathbun, Enrico Marchesini, Xiang Zhi Tan "
venue: TBD
year: 2026
paper_url: "TBD"
type: "conference"
---
Multi-agent reinforcement learning (MARL) in real-world use cases may need to adapt to external natural language instructions that interrupt ongoing behavior and conflict with long-horizon objectives. However, conditioning rewards on instructions introduces a fundamental failure mode as Bellman updates couple value estimates across instruction contexts, leading to inconsistent values when instructions interrupt macro-actions. We propose \textit{Macro-Action Value Correction for Instruction Compliance} (MAVIC), which corrects Bellman backups at instruction boundaries by correcting the incoming instruction objective and restoring the continuation value under the current objective. Unlike reward shaping, MAVIC modifies the bootstrapping target itself, enabling consistent value estimation under stochastic instruction switching within a unified policy. We provide theoretical analysis and an actor-critic implementation, and show that MAVIC achieves high instruction compliance while preserving base task performance in increasingly complex cooperative multi-agent environments.
