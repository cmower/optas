import abc
import time
import yaml
import functools
import casadi as cs
from scipy import interpolate


class Manager(abc.ABC):

    """Base manager class."""

    def __init__(self, config_filename=None, record_solver_perf=False):
        self.reset_manager()
        self.config_filename = config_filename
        self.record_solver_perf = record_solver_perf
        self.config = self._load_configuration(self.config_filename)
        self.solver = self.setup_solver()
        self.solve = self._initialize_solve_method()

    def reset_manager(self):
        self.num_solves = 0  # number of times _solve is called
        self.solver_duration = None
        self.solution = None

    def _load_configuration(self, filename):
        config = {}
        if filename:
            with open(filename, "rb") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
        return config

    def _initialize_solve_method(self):
        if self.record_solver_perf:
            solve = self._solve_and_time
        else:
            solve = self._solve
        return solve

    def _solve(self):
        """Solves the optimization problem."""
        self.solution = self.solver.solve()
        self.num_solves += 1

    def _solve_and_time(self):
        """Solves the optimization problem and times."""
        t0 = time.perf_counter()
        self._solve()
        t1 = time.perf_counter()
        self.solver_duration = t1 - t0

    def get_solver_duration(self):
        """Returns the duration of the solver."""
        return self.solver_duration

    def is_first_solve(self):
        """True when the solver is being run for the first time."""
        return self.num_solves == 0

    ####################################################################################
    ## Abstract methods

    @abc.abstractmethod
    def setup_solver(self) -> None:
        """Setup the optimization solver."""
        pass

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """True when manager is ready to use."""
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the optimization problem."""
        pass

    @abc.abstractmethod
    def get_target(self) -> cs.DM:
        """Return the target from the solution. For example, the target joint states."""
        pass


class ROSManager(Manager):
    # Dictionary that defines the state listener. This is used to
    # setup subscribers that listen to the input that defines the
    # parameters for the solver. The key should be the topic name, and
    # the value should be the message type. E.g.
    #     "joint_states": JointState
    #
    state_listener = {}

    def __init__(self, rosapi, rosver, config_filename, record_solver_perf=False):
        super().__init__(
            config_filename=config_filename, record_solver_perf=record_solver_perf
        )
        self.rosapi = rosapi
        self.rosver = rosver

        # Setup target publisher
        from std_msgs.msg import Float64MultiArray  # available in ros1/2

        self.Float64MultiArray = Float64MultiArray
        self.target_pub = self._setup_target_publisher()

        # Setup publishers/subscriber
        self.msgs = (
            {}
        )  # for storing messages from subscribers setup in create_state_listener
        self.create_state_listener()  # user defined method

    def _setup_target_publisher(self):
        if self.rosver == 1:
            target_pub = self.rosapi.Publisher(
                "target", Float64MultiArray, queue_size=10
            )
        elif self.rosver == 2:
            target_pub = self.rosapi.create_publisher(
                self.Float64MultiArray, "target", 10
            )
        else:
            raise ValueError(f"Did not recognize ros version, given '{self.rosver}'")
        return target_pub

    def add_subscriber(self, topic_name, msg_type):
        self.msgs[topic_name] = None
        callback = functools.partial(self._callback, topic_name=topic_name)
        if self.rosver == 1:
            self.rosapi.Subscriber(topic_name, msg_type, callback)
        elif self.rosver == 2:
            self.rosapi.create_subscription(msg_type, topic_name, callback, 10)
        else:
            raise ValueError(f"Did not recognize ros version, given '{self.rosver}'")

    def _callback(self, msg, topic_name):
        self.msgs[topic_name] = msg

    def get_state(self, topic_name):
        return self.msgs[topic_name]

    def is_ready(self):
        return all(m is not None for m in self.msgs.values())

    def publish_target(self, target: cs.DM):
        msg = self.Float64MultiArray(data=target.toarray().flatten().tolist())
        self.target_pub.publish(msg)

    def create_state_listener(self):
        for topic_name, msg_type in self.state_listener.items():
            if topic_name == "tf2":
                raise NotImplementedError(
                    "Listening to frames from the tf2 library has not yet been implemented."
                )
            self.add_subscriber(topic_name, msg_type)


class ROSController(ROSManager):
    def __init__(
        self, rosapi, rosver, config_filename, hz, urdf_string, record_solver_perf=False
    ):
        self.urdf_string = urdf_string
        self.hz = hz
        super().__init__(
            rosapi, rosver, config_filename, record_solver_perf=record_solver_perf
        )

    def __call__(self):
        self.reset()
        self.solve()
        self.publish_target(self.get_target())


class ROSPlanner(ROSManager):
    def __init__(self, rosapi, rosver, config_filename, record_solver_perf=False):
        self.duration = None
        super().__init__(
            rosapi, rosver, config_filename, record_solver_perf=record_solver_perf
        )

    def set_duration(self, duration: float) -> None:
        self.duration = duration

    def get_duration(self) -> None:
        return self.duration

    def plan(self) -> interpolate.interp1d:
        self.reset()
        self.solve()
        duration = self.get_duration()
        solution = self.get_target()
        return self.solver.interpolate(solution, duration, **interp_kwargs)
