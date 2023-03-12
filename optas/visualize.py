import os

import vtk
from vtkmodules.vtkFiltersSources import (
    vtkCylinderSource,
    vtkSphereSource,
    vtkCubeSource,
)

import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from urdf_parser_py.urdf import Mesh, Cylinder, Sphere

from .spatialmath import *


class Visualizer:
    def __init__(
        self,
        window_size=[1440, 810],
        background_color=[0, 0, 0],
        camera_position=[2, 2, 2],
        camera_focal_point=[0, 0, 0],
        camera_view_up=[0, 0, 1],
        quit_after_delay=None,
    ):
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(*background_color)
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(int(window_size[0]), int(window_size[1]))

        self.camera = self.ren.GetActiveCamera()
        self.camera.SetPosition(*camera_position)
        self.camera.SetFocalPoint(*camera_focal_point)
        self.camera.SetViewUp(*camera_view_up)

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        self.actors = []

        if isinstance(quit_after_delay, float):
            assert quit_after_delay > 0.0, "the delay must be positive"
            quit_after_delay_ms = int(quit_after_delay * 1000.0)
            timer_id = self.iren.CreateOneShotTimer(quit_after_delay_ms)
            self.iren.AddObserver("TimerEvent", self.close, timer_id)

    # def append_actors(self, *actors):
    #     for actor in actors:
    #         if isinstance(actor, list):
    #             self.actors += actor  # assume actor is a list of actors
    #         else:
    #             self.actors.append(actor)  # assume actor is a single actor

    #
    # Set/convert helper methods
    #

    @staticmethod
    def cvt_orientation_to_rotation_matrix(orientation, euler_seq, euler_degrees):
        orientation = orientation.toarray().flatten().tolist()
        if len(orientation) == 4:
            R = Rot.from_quat(orientation)
        elif len(orientation) == 3:
            R = Rot.from_euler(euler_seq, orientation, degrees=euler_degrees)
        else:
            raise ValueError(
                f"length for orientation is incorrect, expected 3 or 4, got {len(orientation)}"
            )
        return R.as_matrix()

    @staticmethod
    def set_tf(actor, position, orientation, euler_seq, euler_degrees):
        R = Visualizer.cvt_orientation_to_rotation_matrix(
            orientation, euler_seq, euler_degrees
        )
        tf = rt2tr(R, position)
        Visualizer.set_transformation(actor, tf)

    @staticmethod
    def set_transformation(actor, tf):
        tf = tf.toarray().flatten().tolist()
        transform = vtk.vtkTransform()
        transform.SetMatrix(tf)
        actor.SetUserTransform(transform)

    @staticmethod
    def set_rgba(actor, rgb, alpha):
        if rgb is not None:
            rgb = rgb.toarray().flatten().tolist()
            assert len(rgb) == 3, f"rgb is incorrect length, got {len(rgb)} expected 3"
            actor.GetProperty().SetColor(*rgb)

        assert (
            0.0 <= alpha <= 1.0
        ), f"the scalar alpha must be in the range [0, 1], got {alpha}"
        actor.GetProperty().SetOpacity(alpha)

    @staticmethod
    def set_translation(actor, position):
        position = position.toarray().flatten().tolist()
        transform = vtk.vtkTransform()
        transform.Translate(position)
        actor.SetUserTransform(transform)

    #
    # Drawing methods
    #

    @arrayify_args
    def line(
        self,
        start=[0.0, 0.0, 0.0],
        end=[1.0, 0.0, 0.0],
        rgb=None,
        alpha=1.0,
        linewidth=1.0,
    ):
        start = start.toarray().flatten().tolist()
        end = end.toarray().flatten().tolist()

        points = vtk.vtkPoints()
        points.InsertNextPoint(*start)
        points.InsertNextPoint(*end)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, 0)
        line.GetPointIds().SetId(0, 1)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        actor.GetProperty().SetLineWidth(linewidth)

        self.set_rgba(actor, rgb, alpha)
        self.actors.append(actor)

        return actor

    @arrayify_args
    def sphere(
        self,
        radius=1.0,
        position=[0.0, 0.0, 0.0],
        rgb=None,
        alpha=1.0,
        theta_resolution=20,
        phi_resolution=20,
    ):
        sphere = vtkSphereSource()
        sphere.SetRadius(radius)
        sphere.SetThetaResolution(theta_resolution)
        sphere.SetPhiResolution(phi_resolution)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.set_translation(actor, position)
        self.set_rgba(actor, rgb, alpha)
        self.actors.append(actor)

        return actor

    @arrayify_args
    def sphere_traj(
        self,
        position_traj,
        radius=1.0,
        rgb=None,
        theta_resolution=20,
        phi_resolution=20,
        alpha_spec=None,
    ):
        default_alpha_spec = {"style": "A"}
        if alpha_spec is None:
            alpha_spec = default_alpha_spec.copy()

        n = position_traj.shape[1]

        if alpha_spec["style"] == "A":
            alpha_min = alpha_spec.get("alpha_min", 0.1)
            alpha_max = alpha_spec.get("alpha_max", 1.0)
            alphas = np.linspace(alpha_min, alpha_max, n).tolist()
        elif alpha_spec["style"] == "B":
            alpha_min = alpha_spec.get("alpha_min", 0.1)
            alpha_max = alpha_spec.get("alpha_max", 1.0)
            alphas = [alpha_min] * (n - 1) + [alpha_max]
        elif alpha_spec["style"] == "C":
            alpha_start = alpha_spec.get("alpha_start", 1.0)
            alpha_mid = alpha_spec.get("alpha_mid", 0.1)
            alpha_end = alpha_spec.get("alpha_end", 1.0)
            alphas = [alpha_start] + [alpha_mid] * (n - 2) + [alpha_end]

        actors = []
        for i, alpha in enumerate(alphas):
            actors.append(
                self.sphere(
                    radius=radius,
                    position=position_traj[:, i],
                    rgb=rgb,
                    alpha=alpha,
                    theta_resolution=theta_resolution,
                    phi_resolution=phi_resolution,
                )
            )

        return actors

    @arrayify_args
    def box(
        self,
        scale=[1, 1, 1],
        rgb=None,
        alpha=1.0,
        position=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
        euler_seq="xyz",
        euler_degrees=False,
    ):
        scale = scale.toarray().flatten().tolist()

        cube = vtk.vtkCubeSource()
        cube.SetBounds(
            -0.5 * scale[0],
            0.5 * scale[0],
            -0.5 * scale[1],
            0.5 * scale[1],
            -0.5 * scale[2],
            0.5 * scale[2],
        )

        # Create a vtkPolyDataMapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())

        # Create a vtkActor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.set_tf(actor, position, orientation, euler_seq, euler_degrees)
        self.set_rgba(actor, rgb, alpha)
        self.actors.append(actor)

        return actor

    @arrayify_args
    def cylinder(
        self,
        radius=1.0,
        height=1.0,
        rgb=None,
        alpha=1.0,
        resolution=20,
        position=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
        euler_seq="xyz",
        euler_degrees=False,
    ):
        cylinder = vtkCylinderSource()
        cylinder.SetRadius(radius)
        cylinder.SetHeight(height)
        cylinder.SetResolution(resolution)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        self.set_tf(actor, position, orientation, euler_seq, euler_degrees)
        self.set_rgba(actor, rgb, alpha)
        self.actors.append(actor)

        return actor

    @arrayify_args
    def cylinder_urdf(
        self,
        radius=1.0,
        height=1.0,
        rgb=None,
        alpha=1.0,
        resolution=20,
        position=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
        euler_seq="xyz",
        euler_degrees=False,
    ):
        cylinder = vtkCylinderSource()
        cylinder.SetRadius(radius)
        cylinder.SetHeight(height)
        cylinder.SetResolution(resolution)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        R = self.cvt_orientation_to_rotation_matrix(
            orientation, euler_seq, euler_degrees
        )
        tf_0 = r2t(Rot.from_euler("x", 90, degrees=True).as_matrix())
        tf_1 = rt2tr(R, position)
        tf = tf_1 @ tf_0

        self.set_transformation(actor, tf)
        self.set_rgba(actor, rgb, alpha)
        self.actors.append(actor)

        return actor

    @arrayify_args
    def text(
        self,
        msg="Hello, world!",
        position=[0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
        rgb=None,
        alpha=1.0,
    ):
        position = position.toarray().flatten().tolist()
        scale = scale.toarray().flatten().tolist()

        # Create a text source to generate the text
        textSource = vtk.vtkTextSource()
        textSource.SetText(msg)
        textSource.Update()

        # Create a mapper to map the text source to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(textSource.GetOutputPort())

        # Create a follower to position the text in 3D space
        follower = vtk.vtkFollower()
        follower.SetMapper(mapper)
        follower.SetPosition(*position)
        follower.SetScale(*scale)
        follower.SetCamera(self.camera)

        self.set_rgba(follower, rgb, alpha)

        self.actors.append(follower)

        return follower

    @arrayify_args
    def link(
        self,
        T=None,
        axis_scale=0.1,
        axis_linewidth=1.0,
        center_radius=0.01,
        center_rgb=None,
        center_alpha=1.0,
        center_theta_resolution=20,
        center_phi_resolution=20,
        axis_alpha=1.0,
    ):
        if T is None:
            T = I4()

        p = transl(T)
        x = T[:3, 0]
        y = T[:3, 1]
        z = T[:3, 2]

        actors = []

        actors.append(
            self.line(
                start=p,
                end=p + axis_scale * x,
                rgb=[1, 0, 0],
                alpha=axis_alpha,
                linewidth=axis_linewidth,
            )
        )

        actors.append(
            self.line(
                start=p,
                end=p + axis_scale * y,
                rgb=[0, 1, 0],
                alpha=axis_alpha,
                linewidth=axis_linewidth,
            )
        )

        actors.append(
            self.line(
                start=p,
                end=p + axis_scale * z,
                rgb=[0, 0, 1],
                alpha=axis_alpha,
                linewidth=axis_linewidth,
            )
        )

        actors.append(
            self.sphere(
                radius=center_radius,
                position=p,
                rgb=center_rgb,
                alpha=center_alpha,
                theta_resolution=center_theta_resolution,
                phi_resolution=center_phi_resolution,
            )
        )

        return actors

    @arrayify_args
    def grid_floor(
        self,
        num_cells=10,
        rgb=None,
        alpha=1.0,
        linewidth=3.0,
        inner_rgb=None,
        inner_alpha=None,
        inner_linewidth=1.0,
        stride=1.0,
        euler=[0, 0, 0],
        euler_seq="xyz",
        euler_degrees=True,
    ):
        num_cells = num_cells.toarray().flatten().astype(int)[0]
        stride = stride.toarray().flatten()[0]

        assert num_cells > 0, "num_cells must be a positive number!"
        assert num_cells % 2 == 0, "num_cells must be even!"

        actors = []

        num_lines = num_cells + 1

        tf_0 = rt2tr(
            I3(),
            [
                -0.5 * float(num_cells) * stride,
                -0.5 * float(num_cells) * stride,
                0.0,
            ],
        )
        tf_1 = r2t(
            Rot.from_euler(
                euler_seq, euler.toarray().flatten(), degrees=euler_degrees
            ).as_matrix()
        )
        tf = tf_1 @ tf_0

        offset = vtk.vtkTransform()
        offset.SetMatrix(tf.toarray().flatten().tolist())

        if inner_rgb is None:
            inner_rgb = rgb

        if inner_alpha is None:
            inner_alpha = alpha

        for i in range(num_cells):
            actor = self.line(
                start=[(float(i) + 0.5) * stride, 0, 0],
                end=[(float(i) + 0.5) * stride, float(num_cells) * stride, 0],
                linewidth=inner_linewidth,
                alpha=inner_alpha,
                rgb=inner_rgb,
            )
            actor.SetUserTransform(offset)
            actors.append(actor)

            actor = self.line(
                start=[0, (float(i) + 0.5) * stride, 0],
                end=[float(num_cells) * stride, (float(i) + 0.5) * stride, 0],
                linewidth=inner_linewidth,
                alpha=inner_alpha,
                rgb=inner_rgb,
            )
            actor.SetUserTransform(offset)
            actors.append(actor)

        for i in range(num_lines):
            actor = self.line(
                start=[float(i) * stride, 0, 0],
                end=[float(i) * stride, float(num_cells) * stride, 0],
                rgb=rgb,
                alpha=alpha,
                linewidth=linewidth,
            )
            actor.SetUserTransform(offset)
            actors.append(actor)

            actor = self.line(
                start=[0, float(i) * stride, 0],
                end=[float(num_cells) * stride, float(i) * stride, 0],
                rgb=rgb,
                alpha=alpha,
                linewidth=linewidth,
            )
            actor.SetUserTransform(offset)
            actors.append(actor)

        return actors

    @arrayify_args
    def obj(
        self,
        obj_filename,
        png_texture_filename=None,
        position=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
        euler_seq="xyz",
        euler_degrees=False,
    ):
        # # Create a renderer, render window, and interactor
        # renderer = vtk.vtkRenderer()
        # render_window = vtk.vtkRenderWindow()
        # render_window.AddRenderer(renderer)
        # interactor = vtk.vtkRenderWindowInteractor()
        # interactor.SetRenderWindow(render_window)

        # Read the .obj file
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_filename)
        reader.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if isinstance(png_texture_filename, str):
            # Create a texture from the png file
            texture = vtk.vtkTexture()
            texture_image = vtk.vtkPNGReader()
            texture_image.SetFileName(png_texture_filename)
            texture_image.Update()
            texture.SetInputConnection(texture_image.GetOutputPort())

            actor.SetTexture(texture)

        self.set_tf(actor, position, orientation, euler_seq, euler_degrees)

        self.actors.append(actor)

        return actor

    @arrayify_args
    def stl(
        self,
        filename,
        scale=None,
        rgb=None,
        alpha=1.0,
        position=[0.0, 0.0, 0.0],
        orientation=[0.0, 0.0, 0.0],
        euler_seq="xyz",
        euler_degrees=False,
    ):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(filename)

        if scale is None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
        else:
            scale = scale.toarray().flatten().tolist()

            transform = vtk.vtkTransform()
            transform.Scale(*scale)

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetTransform(transform)
            transformFilter.SetInputConnection(reader.GetOutputPort())

            # Visualize the object
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transformFilter.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

        self.set_tf(actor, position, orientation, euler_seq, euler_degrees)
        self.set_rgba(actor, rgb, alpha)
        self.actors.append(actor)

        return actor

    def robot(
        self,
        robot_model,
        q=None,
        alpha=1.0,
        show_links=False,
        link_axis_scale=0.2,
        link_axis_linewidth=1.0,
        link_center_radius=0.01,
        link_center_rgb=None,
        link_center_alpha=None,
        display_link_names=False,
        link_names_scale=[0.005, 0.005, 0.005],
        link_names_rgb=[1, 1, 1],
        link_names_alpha=1.0,
    ):
        if link_center_alpha is None:
            link_center_alpha = alpha

        actors = []

        urdf = robot_model.get_urdf()

        material_names = [m.name for m in urdf.materials]

        def get_material_rgba(name):
            try:
                idx = material_names.index(name)
            except ValueError:
                return None
            material = urdf.materials[idx]
            if material.color is not None:
                return material.color.rgba

        if q is None:
            q_user_input = [0.0] * robot_model.ndof
        else:
            q_user_input = cs.vec(q)

        # Setup functions to compute visual origins in global frame
        q = cs.SX.sym("q", robot_model.ndof)
        link_tf = {}
        visual_tf = {}
        for urdf_link in urdf.links:
            name = urdf_link.name

            lnk_tf = robot_model.get_global_link_transform(urdf_link.name, q)
            link_tf[name] = cs.Function(f"link_tf_{name}", [q], [lnk_tf])

            xyz, rpy = robot_model.get_link_visual_origin(urdf_link)
            visl_tf = rt2tr(rpy2r(rpy), xyz)

            tf = lnk_tf @ visl_tf
            visual_tf[name] = cs.Function(f"visual_tf_{name}", [q], [tf])

        for urdf_link in urdf.links:
            geometry = urdf_link.visual.geometry
            tf = visual_tf[urdf_link.name](q_user_input).toarray()
            position = tf[:3, 3].flatten().tolist()
            orientation = Rot.from_matrix(tf[:3, :3]).as_quat().tolist()

            if show_links:
                actors += self.link(
                    tf,
                    axis_scale=link_axis_scale,
                    axis_linewidth=link_axis_linewidth,
                    center_radius=link_center_radius,
                    center_rgb=link_center_rgb,
                    center_alpha=link_center_alpha,
                )

            if display_link_names:
                actors.append(
                    self.text(
                        msg=urdf_link.name,
                        position=position,
                        scale=link_names_scale,
                        rgb=link_names_rgb,
                        alpha=link_names_alpha,
                    )
                )

            if urdf_link.visual is None:
                continue

            material = urdf_link.visual.material
            rgb = None
            if isinstance(material.name, str) and material.name in material_names:
                rgba = get_material_rgba(urdf_link.visual.material.name)
                rgb = rgba[:3]

            if isinstance(geometry, Mesh):
                if geometry.filename.lower().endswith(".stl"):
                    filename = geometry.filename

                    if not os.path.exists(filename):
                        relpath = robot_model.get_urdf_dirname()
                        filename = os.path.join(relpath, filename)

                    scale = None
                    if geometry.scale is not None:
                        scale = geometry.scale

                    actors.append(
                        self.stl(
                            filename,
                            scale=scale,
                            rgb=rgb,
                            alpha=alpha,
                            position=position,
                            orientation=orientation,
                            euler_seq="xyz",
                            euler_degrees=True,
                        )
                    )

            elif isinstance(geometry, Sphere):
                actors.append(
                    self.sphere(
                        radius=geometry.radius,
                        position=position,
                        rgb=rgb,
                        alpha=alpha,
                        theta_resolution=20,
                        phi_resolution=20,
                    )
                )

            elif isinstance(geometry, Cylinder):
                actors.append(
                    self.cylinder_urdf(
                        radius=geometry.radius,
                        height=geometry.length,
                        position=position,
                        orientation=orientation,
                        rgb=rgb,
                        alpha=alpha,
                    )
                )

        return actors

    def robot_traj(
        self,
        robot_model,
        Q,
        alpha_spec=None,
        show_links=False,
        link_axis_scale=0.2,
        link_axis_linewidth=1.0,
        link_center_rgb=1,
        link_center_alpha=None,
        link_center_radius=0.01,
        display_link_names=False,
        link_names_scale=[0.01, 0.01, 0.01],
        link_names_rgb=[1, 1, 1],
        link_names_alpha=1.0,
    ):
        default_alpha_spec = {"style": "A"}
        if alpha_spec is None:
            alpha_spec = default_alpha_spec.copy()

        actors = []
        n = Q.shape[1]

        if alpha_spec["style"] == "A":
            alpha_min = alpha_spec.get("alpha_min", 0.1)
            alpha_max = alpha_spec.get("alpha_max", 1.0)
            alphas = np.linspace(alpha_min, alpha_max, n).tolist()
        elif alpha_spec["style"] == "B":
            alpha_min = alpha_spec.get("alpha_min", 0.1)
            alpha_max = alpha_spec.get("alpha_max", 1.0)
            alphas = [alpha_min] * (n - 1) + [alpha_max]
        elif alpha_spec["style"] == "C":
            alpha_start = alpha_spec.get("alpha_start", 1.0)
            alpha_mid = alpha_spec.get("alpha_mid", 0.1)
            alpha_end = alpha_spec.get("alpha_end", 1.0)
            alphas = [alpha_start] + [alpha_mid] * (n - 2) + [alpha_end]

        for i, alpha in enumerate(alphas):
            actors += self.robot(
                robot_model,
                Q[:, i],
                alpha,
                show_links,
                link_axis_scale,
                link_axis_linewidth,
                link_center_rgb,
                link_center_alpha,
                link_center_radius,
                display_link_names,
                link_names_scale,
                link_names_rgb,
                link_names_alpha,
            )

        return actors

    def start(self):
        for actor in self.actors:
            self.ren.AddActor(actor)
        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def close(self, obj, event):
        self.renWin.Finalize()
        self.iren.TerminateApp()
