---
title: "OpTaS: An Optimization-based Task Specification Python Framework For Robot Control and Planning"
tags:
- robotics
- control
- planning
- task and motion planning (TAMP)
- task specification
authors:
- name: Christopher E. Mower
  orcid: 0000-0002-3929-9391
  affiliation: 1
- name: João Moura
  affiliation: 2
- name: Martin Huber
  affiliation: 1
- name: Nazanin Zamani Behabadi
  affiliation: 3
- name: Sethu Vijayakumar
  affiliation: 2, 4
- name: Tom Vercauteren
  affiliation: 1
- name: Christos Bergeles
  affiliation: 1
affiliations:
- name: King's College London, United Kingdom
  index: 1
- name: University of Edinburgh, United Kingdom
  index: 2
- name: Unaffiliated, United Kingdom
  index: 3
- name: The Alan Turing Institute, United Kingdom
  index: 4
date: 30 December 2022
bibliography: paper.bib
---

# Summary

This paper presents OpTaS and OpTaS-ROS:
OpTaS is a task specification Python library for Trajectory Optimization (TO), Task and Motion Planning (TAMP), and Model Predictive Control (MPC) in robotics;
and OpTaS-ROS is an interface layer to the Robot Operating System [@Quigley09], allowing a user to programatically (un)load controllers/planners via a dedicated service.
Both TO and MPC are increasingly receiving interest in optimal control and in particular handling dynamic environments.

While a flurry of software libraries exists to handle such problems, they either provide interfaces that are limited to a specific problem formulation (e.g. TracIK, CHOMP), or are large and statically specify the problem  in configuration files (e.g. EXOTica, eTaSL).
OpTaS, on the other hand, allows a user to specify custom nonlinear and even discrete constrained problem formulations in a single Python script allowing the controller parameters to be modified during execution.
The library provides interface to several open source and commercial solvers (e.g. IPOPT, SNOPT, KNITRO, SciPy, OSQP, among others) to facilitate integration with established workflows in robotics.
Further benefits of OpTaS are highlighted through a thorough comparison with common libraries.
An additional key advantage of OpTaS is the ability to define optimal control tasks in the joint space, task space, or indeed simultaneously.

# Acknowledgments

This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101016985 (FAROS project).
This work was supported by core funding from the Wellcome/EPSRC [WT203148/Z/16/Z; NS/A000049/1].
Tom Vercauteren is supported by a Medtronic / RAEng Research Chair [RCSRF1819\textbackslash7\textbackslash34].
For the purpose of open access, the authors have applied a CC BY public copyright licence to any Author Accepted Manuscript version arising from this submission.

# References
