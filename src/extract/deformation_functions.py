import numpy as np


class DeformationFunctions:
    def __init__(self,w=0.1):
        self.w = w # weight for the strength of deformation functions

    def apply_deformation(self, feature_template, trajectory, pos_obj):
        print(feature_template)
        if feature_template == "obj_distance_decrease":
            if pos_obj is not None: 
                return self.obj_distance_decrease(trajectory, pos_obj)
            else: 
                raise AssertionError("Position of the object is not given")
        elif feature_template == "obj_distance_increase":
            if pos_obj is not None: 
                return self.obj_distance_increase(trajectory, pos_obj)
            else: 
                raise AssertionError("Position of the object is not given")
        elif feature_template == "z_cart_decrease":
            direction = np.array([0.0, 0.0, -1.0])
            return self.cart_change(trajectory, direction)
        elif feature_template == "z_cart_increase":
            direction = np.array([0.0, 0.0, 1.0])
            return self.cart_change(trajectory, direction)
        elif feature_template == "y_cart_decrease":
            direction = np.array([0.0, -1.0, 0.0])
            return self.cart_change(trajectory, direction)
        elif feature_template == "y_cart_increase":
            direction = np.array([0.0, 1.0, 0.0])
            return self.cart_change(trajectory, direction)
        elif feature_template == "x_cart_decrease":
            direction = np.array([-1.0, 0.0, 0.0])
            return self.cart_change(trajectory, direction)
        elif feature_template == "x_cart_increase":
            direction = np.array([1.0, 0.0, 0.0])
            return self.cart_change(trajectory, direction)
        else:
            raise NotImplementedError(
                "Only the following deformation functions are available: obj_distance_decrease, obj_distance_increase, z_cart_decrease, y_cart_decrease, x_cart_decrease, z_cart_increase, y_cart_increase, x_cart_increase"
            )

    def obj_distance_decrease(self, trajectory, pos_obj, r=0.3):
        """
        Deformation function δ for 'obj_distance_decrease'.

        Args:
            trajectory (list of list): Original trajectory, each waypoint [x, y, z, vel]
            pos_obj (list or np.array): Position of the object as [x, y, z]
            'r': Radius of deformation (default is 0.3 if not provided)

        Returns:
            list of list: Deformed trajectory, same structure as trajectory

            
        From Extract: 
        For scene-specific features, i.e. object distance features,
        the force exerted is dependent on the object position opos
        and a radius of deformation r. In our experiments, we set
        r = 0.3, which was determined empirically. We note that r
        can be set adaptively based on environmental constraints and
        human preferences but this will be part of our future work.
        For waypoints within the radius of deformation r from the
        object position opos, a force is applied on the waypoints in
        the direction of the distance vector between the waypoint and
        the object. The force is 0 for other waypoints.

        For scene-independent features, a force is exerted on all
        waypoints of the trajectory, where the direction of the force is
        dependent on the feature.
        The trajectory is deformed based on the force calculated on
        each waypoint : δ = ξ0 + wF, where the weight w changes
        the magnitude of the deformation. We empirically determined
        the value of w to be a constant of 1.0 in our experiments


        """
        opos = np.array(pos_obj)

        new_traj = []
        for waypoint in trajectory:
            
            pos = np.array(waypoint[:3])
            vel = waypoint[3]

            # Compute distance and direction vector
            dist_vec = opos - pos
            dist = np.linalg.norm(dist_vec)

            if dist < r and dist > 1e-6 and dist>0.01 :  # Within radius and avoid division by zero, don't come too near to the objects
                # Apply force proportional to distance (pull toward object)
                direction = dist_vec / dist
                force_magnitude = (r - dist) / r  # Linearly decays to 0 at r
                F = force_magnitude * direction  # Force vector
                new_pos = pos + self.w * F  # Apply weighted force
            else:
                new_pos = pos  # No change

            new_traj.append(list(new_pos) + [vel])  # Keep velocity unchanged

        return new_traj

    def obj_distance_increase(self, trajectory, pos_obj, r=0.3):
        """
        Deformation function δ for 'obj_distance_increase'.

        Args:
            trajectory (list of list): Original trajectory, each waypoint [x, y, z, vel]
            pos_obj (list or np.array): Position of the object as [x, y, z]
            r (float): Radius of deformation (default is 0.3)

        Returns:
            list of list: Deformed trajectory, same structure as trajectory
        """
        opos = np.array(pos_obj)

        new_traj = []
        for waypoint in trajectory:
            pos = np.array(waypoint[:3])
            vel = waypoint[3]

            # Compute distance and direction vector
            dist_vec = pos - opos  # Direction: away from object
            dist = np.linalg.norm(dist_vec)

            if dist < r and dist > 1e-6  and dist>0.01:  # Within radius and avoid division by zero, avoid colliding with the object
                # Apply force proportional to distance (push away from object)
                direction = dist_vec / dist
                force_magnitude = (r - dist) / r  # Linearly decays to 0 at r
                F = force_magnitude * direction  # Force vector
                new_pos = pos + self.w * F  # Apply weighted force
            else:
                new_pos = pos  # No change

            new_traj.append(list(new_pos) + [vel])  # Keep velocity unchanged

        return new_traj

    def cart_change(self, trajectory, direction):
        """
        Deformation function δ for 'z_cart_decrease'.

        Applies a downward force to decrease the z-coordinate of all waypoints,
        with weaker force near the start and stronger toward the end.

        Args:
            trajectory (list of list): Original trajectory, each waypoint [x, y, z, vel]
        Returns:
            list of list: Deformed trajectory, same structure as trajectory

        direction is framed as np.array([0.0, 0.0, -1.0]) for -z change
        """
        n = len(trajectory)

        new_traj = []
        for i, waypoint in enumerate(trajectory):
            pos = np.array(waypoint[:3])
            vel = waypoint[3]

            # Compute scaling factor: 0 at start, 1 at end
            scale = i / (n - 1) if n > 1 else 1.0

            # Apply downward force scaled by position in trajectory
            # This is not mentioned in the paper explicilty but based on the results in the paper, this seems correct
            F = scale * direction
            new_pos = pos + self.w * F

            new_traj.append(list(new_pos) + [vel])  # Keep velocity unchanged

        return new_traj
