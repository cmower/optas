import os
import optas
import pytest
import pathlib


class Test_OptimizationBuilder:
    def test_init_null_input(self):
        with pytest.raises(TypeError):
            builder = optas.OptimizationBuilder()

    def test_init_no_models(self):
        T = 10
        builder = optas.OptimizationBuilder(T)
        assert builder.T == T
        assert len(builder._models) == 0
        assert builder.derivs_align == False

    def test_init_models(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        assert len(builder._models) == 1

        map_sx_container_label_to_expected_numel = {
            "_decision_variables": 30,
            "_parameters": 0,
            "_cost_terms": 0,
            "_lin_eq_constraints": 0,
            "_lin_ineq_constraints": 0,
            "_ineq_constraints": 0,
            "_eq_constraints": 0,
        }

        for label, expected_numel in map_sx_container_label_to_expected_numel.items():
            sx_container = getattr(builder, label)
            assert sx_container.numel() == expected_numel

    def test_get_model_names(self):
        T = 10
        task_model_1 = optas.TaskModel("test1", 3)
        task_model_2 = optas.TaskModel("test2", 4)
        builder = optas.OptimizationBuilder(T, tasks=[task_model_1, task_model_2])

        expected_model_names = ["test1", "test2"]
        model_names = builder.get_model_names()
        assert len(expected_model_names) == len(model_names)
        for name, expected_name in zip(model_names, expected_model_names):
            assert name == expected_name

    def test_get_model_index(self):
        T = 10
        task_model_1 = optas.TaskModel("test1", 3)
        task_model_2 = optas.TaskModel("test2", 4)
        builder = optas.OptimizationBuilder(T, tasks=[task_model_1, task_model_2])

        assert builder.get_model_index("test1") == 0
        assert builder.get_model_index("test2") == 1

    def test_get_model(self):
        T = 10
        task_model_1 = optas.TaskModel("test1", 3)
        task_model_2 = optas.TaskModel("test2", 4)
        builder = optas.OptimizationBuilder(T, tasks=[task_model_1, task_model_2])

        assert builder.get_model("test1") is task_model_1
        assert builder.get_model("test2") is task_model_2

    def test_get_model_state(self):
        T = 10
        task_model_1 = optas.TaskModel("test1", 3)
        task_model_2 = optas.TaskModel("test2", 4, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=[task_model_1, task_model_2])

        state = builder.get_model_state("test1", 0)
        assert isinstance(state, optas.SX)
        assert state.shape[0] == 3
        assert state.shape[1] == 1

        state = builder.get_model_state("test2", 0, time_deriv=1)
        assert isinstance(state, optas.SX)
        assert state.shape[0] == 4
        assert state.shape[1] == 1

    def test_get_model_states(self):
        T = 10
        derivs_align = False
        task_model_1 = optas.TaskModel("test1", 3)
        task_model_2 = optas.TaskModel("test2", 4, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(
            T, tasks=[task_model_1, task_model_2], derivs_align=derivs_align
        )

        state = builder.get_model_states("test1")
        assert isinstance(state, optas.SX)
        assert state.shape[0] == 3
        assert state.shape[1] == T

        state = builder.get_model_states("test2", time_deriv=1)
        assert isinstance(state, optas.SX)
        assert state.shape[0] == 4
        assert state.shape[1] == T - 1

        T = 10
        derivs_align = True
        task_model_1 = optas.TaskModel("test1", 3)
        task_model_2 = optas.TaskModel("test2", 4, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(
            T, tasks=[task_model_1, task_model_2], derivs_align=derivs_align
        )

        state = builder.get_model_states("test1")
        assert isinstance(state, optas.SX)
        assert state.shape[0] == 3
        assert state.shape[1] == T

        state = builder.get_model_states("test2", time_deriv=1)
        assert isinstance(state, optas.SX)
        assert state.shape[0] == 4
        assert state.shape[1] == T

    def test_get_model_parameters(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path, param_joints=["joint0"])
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)

        parameters = builder.get_model_parameters(name)
        assert isinstance(parameters, optas.SX)
        assert parameters.shape[0] == 1
        assert parameters.shape[1] == T

    def test_get_model_parameter(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path, param_joints=["joint0"])
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)

        parameters = builder.get_model_parameter(name, 0)
        assert isinstance(parameters, optas.SX)
        assert parameters.shape[0] == 1
        assert parameters.shape[1] == 1

    def test_get_robot_states_and_parameters(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path, param_joints=["joint0"])
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)

        states_and_parameters = builder.get_robot_states_and_parameters(name)
        assert isinstance(states_and_parameters, optas.SX)
        assert states_and_parameters.shape[0] == 3
        assert states_and_parameters.shape[1] == T

    def test_get_model_robot_states_and_parameters_error(self):
        with pytest.raises(AssertionError):
            T = 10
            task_model = optas.TaskModel("test", 3)
            builder = optas.OptimizationBuilder(T, tasks=task_model)
            states_and_parameters = builder.get_robot_states_and_parameters("test")

    def test_x(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)

        x = builder._x()
        assert isinstance(x, optas.SX)
        assert optas.vec(x).shape[0] == 30

    def test_p(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path, param_joints=["joint0"])
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)

        p = builder._p()
        assert isinstance(p, optas.SX)
        assert optas.vec(p).shape[0] == 10

    def test_is_linear_in_x(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        x = builder._x()
        y = 2.0 * x
        assert builder._is_linear_in_x(y) == True

    def test_cost(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder.get_model_states("test")
        builder.add_cost_term("test_cost", optas.sumsqr(states))
        cost = builder._cost()
        assert isinstance(cost, optas.SX)
        assert cost.shape[0] == 1
        assert cost.shape[1] == 1

    def test_is_cost_quadratic(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder.get_model_states("test")
        builder.add_cost_term("test_cost", optas.sumsqr(states))

        assert builder.is_cost_quadratic() == True

        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder.get_model_states("test")
        builder.add_cost_term("test_cost", optas.cos(optas.sumsqr(states)))

        assert builder.is_cost_quadratic() == False

    # methods: add_decision_variables, add_parameters, and
    # add_cost_term already tested in methods above.

    def test_add_geq_inequality_constraint(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder._x()
        g_lin = states - 2.0
        builder.add_geq_inequality_constraint("test_con1", g_lin)
        g_nlin = optas.cos(states[0]) + 1.0
        builder.add_geq_inequality_constraint("test_con2", g_nlin)

        assert builder._lin_ineq_constraints.numel() == 30
        assert builder._ineq_constraints.numel() == 1

    def test_add_leq_inequality_constraint(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder._x()
        g_lin = states - 2.0
        builder.add_leq_inequality_constraint("test_con1", g_lin)
        g_nlin = optas.cos(states[0]) + 1.0
        builder.add_leq_inequality_constraint("test_con2", g_nlin)

        assert builder._lin_ineq_constraints.numel() == 30
        assert builder._ineq_constraints.numel() == 1

    def test_add_bound_inequality_constraint(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder._x()
        g_lin = states - 2.0
        builder.add_bound_inequality_constraint("test_con1", 1.0, g_lin, 10.0)
        g_nlin = optas.cos(states[0]) + 1.0
        builder.add_bound_inequality_constraint("test_con2", 1.0, g_nlin, 10.0)

        assert builder._lin_ineq_constraints.numel() == 2 * 30
        assert builder._ineq_constraints.numel() == 2 * 1

    def test_add_equality_constraint(self):
        T = 10
        task_model = optas.TaskModel("test", 3)
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        states = builder._x()
        g_lin = states - 2.0
        builder.add_equality_constraint("test_con1", g_lin)
        g_nlin = optas.cos(states[0]) + 1.0
        builder.add_equality_constraint("test_con2", g_nlin)

        assert builder._lin_eq_constraints.numel() == 30
        assert builder._eq_constraints.numel() == 1

    def test_integrate_model_states(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model)
        builder.integrate_model_states("test", 1, 1)
        assert builder._lin_eq_constraints.numel() == 2 * (T - 1)

        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        builder.integrate_model_states("test", 1, 1)
        assert builder._lin_eq_constraints.numel() == 2 * (T - 1)

    def test_enforce_model_limits(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path)
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)
        builder.enforce_model_limits(name)
        assert builder._lin_ineq_constraints.numel() == 60

    def test_initial_configuration(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path)
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)
        builder.initial_configuration(name)
        assert builder._lin_eq_constraints.numel() == 3

    def test_fix_configuration(self):
        cwd = pathlib.Path(
            __file__
        ).parent.resolve()  # path to current working directory
        urdf_filename = "tester_robot.urdf"
        urdf_path = os.path.join(cwd, urdf_filename)

        model = optas.RobotModel(urdf_filename=urdf_path)
        name = model.get_name()

        T = 10
        builder = optas.OptimizationBuilder(T, robots=model)
        builder.fix_configuration(name)
        assert builder._lin_eq_constraints.numel() == 3

    def test_build_QuadraticCostUnconstrained(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        X = builder.get_model_states("test")
        builder.add_cost_term("test_term", optas.sumsqr(X))
        opt = builder.build()
        assert isinstance(opt, optas.optimization.QuadraticCostUnconstrained)

    def test_build_QuadraticCostLinearConstraints(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        X = builder.get_model_states("test")
        builder.add_cost_term("test_term", optas.sumsqr(X))
        builder.add_bound_inequality_constraint("test_limits", -100, X, 100)
        opt = builder.build()
        assert isinstance(opt, optas.optimization.QuadraticCostLinearConstraints)

    def test_build_QuadraticCostNonlinearConstraints(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        X = builder.get_model_states("test")
        builder.add_cost_term("test_term", optas.sumsqr(X))
        builder.add_bound_inequality_constraint("test_limits", -100, X, 100)
        builder.add_equality_constraint("test_eq_constraints", X[0] * X[1])
        opt = builder.build()
        assert isinstance(opt, optas.optimization.QuadraticCostNonlinearConstraints)

    def test_build_NonlinearCostUnconstrained(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        X = builder.get_model_states("test")
        builder.add_cost_term("test_term1", optas.sumsqr(X))
        builder.add_cost_term("test_term2", optas.cos(X[0] * X[1]))
        opt = builder.build()
        assert isinstance(opt, optas.optimization.NonlinearCostUnconstrained)

    def test_build_NonlinearCostLinearConstraints(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        X = builder.get_model_states("test")
        builder.add_cost_term("test_term1", optas.sumsqr(X))
        builder.add_cost_term("test_term2", optas.cos(X[0] * X[1]))
        builder.add_bound_inequality_constraint("test_limits", -100, X, 100)
        opt = builder.build()
        assert isinstance(opt, optas.optimization.NonlinearCostLinearConstraints)

    def test_build_NonlinearCostNonlinearConstraints(self):
        T = 10
        task_model = optas.TaskModel("test", 2, time_derivs=[0, 1])
        builder = optas.OptimizationBuilder(T, tasks=task_model, derivs_align=True)
        X = builder.get_model_states("test")
        builder.add_cost_term("test_term1", optas.sumsqr(X))
        builder.add_cost_term("test_term2", optas.cos(X[0] * X[1]))
        builder.add_bound_inequality_constraint("test_limits", -100, X, 100)
        builder.add_equality_constraint("test_eq_constraints", X[0] * X[1])
        opt = builder.build()
        assert isinstance(opt, optas.optimization.NonlinearCostNonlinearConstraints)
