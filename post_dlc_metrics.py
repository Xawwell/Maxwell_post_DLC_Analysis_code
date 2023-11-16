#Need to duplicate these for my own DLC output


# -----KALMAN FILTER FUNCS--------------------------------------------------------------
def extract_data_from_dlc_file(self, session) -> None:
        """
        Ingests a H5 file outputted from DLC analysis, body parts, and
        model name. Changing the string in dlc network name maybe necessary if using different model
        type. 
        
        The function saves the body parts tracked by DLC to the tracking data dictionary.

        Args:
            session (obejct): Session settings object data class
        """
        
        # Load DLC Tracking Data
        dlc_tracking_file = glob.glob(os.path.join(session.base_path,session.processed_path, "*.h5"))[0] #Selects the .h5 file in video dir
        self.dlc_output = pd.read_hdf(dlc_tracking_file) #Converts .h5 to pandas
        
        # Load DLC Config from settings dataclass 
        _, dlc_settings_file = get_computer_specific_paths()
        with open(dlc_settings_file) as file: 
            dlc_settings = yaml.safe_load(file)
        
        self.tracking_data_body_parts = {} # init dictionary
        self.tracking_data_body_parts['bodyparts'] = dlc_settings['bodyparts']

        # fix names to make consistent across models
        if 'right_hindpaw' in self.tracking_data_body_parts['bodyparts']:
            self.dlc_output = self.dlc_output.rename(columns={"right_hindpaw": "right_hind_limb"})
            self.tracking_data_body_parts['bodyparts'] = list(map(lambda x: x.replace('right_hindpaw', 'right_hind_limb'), self.tracking_data_body_parts['bodyparts']))
        if 'left_hindpaw' in self.tracking_data_body_parts['bodyparts']:
            self.dlc_output = self.dlc_output.rename(columns={"left_hindpaw": "left_hind_limb"})
            self.tracking_data_body_parts['bodyparts'] = list(map(lambda x: x.replace('left_hindpaw', 'left_hind_limb'), self.tracking_data_body_parts['bodyparts']))
        
        
        logger.info(f"The bodyparts tracked by DLC are: {self.tracking_data_body_parts['bodyparts']}")
        
        self.dlc_network_name = dlc_tracking_file[dlc_tracking_file.find('DLC_resnet'):-3] # This line breaks if different model names are used
        assert self.dlc_network_name, "No DLC name found, has a different model been used?"
        logger.info(f"The DLC network name is: {self.dlc_network_name}")

        return None
    def apply_kalman(self, session) -> None:
        """
        The kalman filter is a recursive algorithm that estimates the state of a system using a sequence of measurements.
        This function requires the tracking data to be in the form of a numpy array with the following dimensions:
        + (2, frames)
        The algorithm works on a single body part and thus needs to be called in a recursive manner.
        """
        
        # Check if kalman tracking data already exists
        if os.path.isfile(os.path.join(session.base_path,session.processed_path, "kalman_tracking_data.pickle")):
            logger.warning("Kalman tracking exists but you've chosen to redo processing")
        
        # Create new kalman tracking data
        logger.info("Creating new kalman tracking data.")
        ldsResults = {}
        
        for i, bodypart in enumerate(self.tracking_data_body_parts['bodyparts']):
            x = self.registered_tracking_data_before_kalman[bodypart][:, 0]
            y = self.registered_tracking_data_before_kalman[bodypart][:, 1]
            xy = np.vstack((x, y))
            
            results = kalmann(xy)
            ldsResults[bodypart] = {"x": results["x"], 
                                    "y": results["y"], 
                                    "likelihood": self.tracking_data_array[:, i, 2],
                                    "xVelocity": results["xVelocity"],
                                    "yVelocity": results["yVelocity"],
                                    "xAccel": results["xAccel"],
                                    "yAccel": results["yAccel"],
                                    }
            
        self.lds_tracking_data = ldsResults
        self.save_kalman(self.lds_tracking_data, session)
        return None
        
    def save_kalman(self, dictionary, session) -> None:
        """
        Save the kalman tracking dictionary to a pickle file contained within the session folder. 
        """
        savePath = os.path.join(session.base_path,session.processed_path, "kalman_tracking_data.pickle")
        with open(savePath, "wb") as dill_file: 
            pickle.dump(dictionary, dill_file)

# -----METRIC COMPUTATION FUNCS--------------------------------------------------------------

    def compute_metrics(self, session):
        # Leaving in session as you in this reference location speed computation
        regionsOI = self.map_regions_of_interest()
        self.compute_avg_mouse_location(regionsOI)
        self.region_tracking_data['hdir'] = self.compute_head_direction()
        self.compute_angle_shelter(session)
        self.compute_angle_barrier(session)
        if len(self.settings.random_points) > 0 :self.compute_angle_random_points(session)
        self.compute_new_average_speed(session)
        self.region_tracking_data['bodyparts'] = self.tracking_data_body_parts['bodyparts'] # Needed for visualization
        # self.compute_speed(session, reference_location=session.video.shelter_location, reference_name=' rel. to shelter')
        
    def map_regions_of_interest(self) -> dict:
        """
        Map regions of body to individual body parts. This function needs to be changed if the list of 
        body parts change in deep lab cut. These are hardcoded regions of interest that need to be mannually mapped to the body parts.
        """
        regionsOI = {
                     'avg_loc': self.tracking_data_body_parts['bodyparts'],
                     'neck_loc': ['left_ear', 'upper_back', 'right_ear'],
                     'upper_body_loc': ['left_shoulder', 'upper_back', 'right_shoulder'],
                     'lower_body_loc': ['left_hind_limb', 'lower_back', 'right_hind_limb', 'tail_base'],
                     'head_loc': ['left_ear', 'right_ear'],
                     'body_loc': ['upper_back', 'lower_back']
                     }
        return regionsOI
            
    def compute_avg_mouse_location(self, regionsOI):
        """
        Compute the average location of the body parts in a region of interest into a dictionary called 
        region tracking data. This is done by taking the mean of the x and y coordinates of the body parts"
        """
        
        self.region_tracking_data = {}
        
        for region in regionsOI.keys():
            x = np.mean([self.lds_tracking_data[bodypart]['x'] for bodypart in regionsOI[region]], axis=0)
            y = np.mean([self.lds_tracking_data[bodypart]['y'] for bodypart in regionsOI[region]], axis=0)
            self.region_tracking_data[region] = np.array([[x, y] for x, y in zip(x, y)])
        
        logger.info("Body points averaged and their positions have been averaged and mapped to regions of interest.")
        
        
    def compute_head_direction(self) -> np.ndarray:
        """
        This function computes the head direction of the mouse. It does this by:
        - taking the difference between the ears
        - taking the archtan2 to compute the angle of the slope
        - rotating that vector such that 0 degrees is pointing towards the door of the rig OR because the line connecting the ears is orthogonal to headirection? 
        - the negative arctan2 is unknown, potentially to do with the origin of the coordinate system?
        - then normalizing the angle so that it stays within the range (-π, π]. As the rotation creates angles less than -π.
        """
        
        # New head direction calculation -----------------------------------------------------------------------------
        hedDelta_x = self.lds_tracking_data['left_ear']['x'] - self.lds_tracking_data['right_ear']['x']
        hedDelta_y = self.lds_tracking_data['left_ear']['y'] - self.lds_tracking_data['right_ear']['y']
        headDirection = - (np.arctan2(hedDelta_y, hedDelta_x) + (np.pi/2)) # Radians
        mask = headDirection < -np.pi # A boolean mask to find all the values less than -pi
        headDirection[mask] = headDirection[mask] + (2*np.pi)
        return headDirection
    
    def compute_angle_shelter(self, session):
        """
        A function to compute the angle between the heading of the mouse and the shelter.
        """

        # calculate body to shelter angle
        # this used to be calculated with self.region_tracking_data['avg_loc']
        self.region_tracking_data['shelter_loc'] = session.shelter_location
        xdist = -self.region_tracking_data['head_loc'][:, 0]+int(np.mean([self.region_tracking_data['shelter_loc'][0][0],self.region_tracking_data['shelter_loc'][1][0]]))
        ydist = -self.region_tracking_data['head_loc'][:, 1]+int(np.mean([self.region_tracking_data['shelter_loc'][0][1],self.region_tracking_data['shelter_loc'][1][1]]))
        # the next line gives you angles that are positive counterclockwise and negative clockwise
        self.region_tracking_data['bod_shelt_dir'] = - np.arctan2(ydist, xdist) # Radians
        bod_shelt_dir = - np.arctan2(ydist, xdist)
        # the next two lines ensure that 0deg is to the right and that 0 to pi is clockwise and 0 to pi is counterclockwise
        self.region_tracking_data['bod_shelt_dir'][bod_shelt_dir<0] = self.region_tracking_data['bod_shelt_dir'][bod_shelt_dir<0] + np.pi
        self.region_tracking_data['bod_shelt_dir'][bod_shelt_dir>0] = self.region_tracking_data['bod_shelt_dir'][bod_shelt_dir>0] - np.pi
        # now add the hdir to get the head shelter angle (from pi to -pi)
        self.region_tracking_data['hdir_shelt'] = np.pi + (-self.region_tracking_data['hdir'] + self.region_tracking_data['bod_shelt_dir'])
        self.region_tracking_data['hdir_shelt'][self.region_tracking_data['hdir_shelt']>np.pi] = self.region_tracking_data['hdir_shelt'][self.region_tracking_data['hdir_shelt']>np.pi] - (2*np.pi)

        logger.info("Shelter angle computed")

    def compute_angle_barrier(self, session):
        """
        A function to compute the angle between the heading of the mouse and the barrier edges.
        """

        if len(session.barrier_time) > 0:
            # initialize variables
            self.region_tracking_data['barrier_loc'] = session.barrier_location
            self.region_tracking_data['hdir_barrier'] = np.empty((len(self.region_tracking_data['avg_loc']),len(self.region_tracking_data['barrier_loc'])))

            for i in np.arange(len(self.region_tracking_data['barrier_loc'])): # calculate body to barrier angle for each edge of barrier
                self.region_tracking_data['hdir_barrier'][:,i] = compute_angle_head_point(self,'barrier_loc',i)
        else:
            self.region_tracking_data['barrier_loc'] = []
        logger.info("Subgoal angles computed")
    
    def compute_angle_random_points(self,session):
        """
        A function to compute the angle between the heading of the mouse and the barrier edges.
        If settings.random_points == 'manual' it will ask you to define random points in the arena
        """

        # ask user to select some extra 'random' points in arena
        if self.settings.random_points == 'manual':
            self.load_arena(session)
            print("Click as many random points as wanted, then space bar when satisfied")
            cv2.namedWindow('where are random points')
            self.clicked_points = []
            cv2.setMouseCallback('where are random points', self.click_click_targets)
            while True:
                cv2.imshow('where are random points', self.arena)
                key = cv2.waitKey(10)
                if key==ord(' '): break # once both points are clicked
                if key == ord('q'): print('quit.'); sys.exit()
            cv2.destroyAllWindows()
            self.region_tracking_data['randP_loc'] = self.clicked_points
        elif self.settings.random_points == 'full_arena':
            size = session.video.height # assuming a square image
            all_posX = []
            all_posY = []
            numpoints = 64
            for i in np.arange(numpoints/2,size,numpoints):
                all_posX = np.append(all_posX,np.arange(numpoints/2,size,numpoints))
                all_posY = np.append(all_posY,np.ones(len(np.arange(numpoints/2,size,numpoints)))*i)
            dist = np.sqrt(((all_posX - size/2)**2) + ((all_posY - size/2)**2))
            all_posX = all_posX[dist<460] # size of arena circle, see register
            all_posY = all_posY[dist<460]
            self.region_tracking_data['randP_loc'] = np.vstack([all_posX,all_posY]).T
        
        # initialize variables
        self.region_tracking_data['hdir_randP'] = np.empty((len(self.region_tracking_data['avg_loc']),len(self.region_tracking_data['randP_loc'])))

        for i in np.arange(len(self.region_tracking_data['randP_loc'])): # calculate body to barrier angle for each edge of barrier
            self.region_tracking_data['hdir_randP'][:,i] = compute_angle_head_point(self,'randP_loc',i)

        logger.info("Random point angles computed")
   
    def click_click_targets(self, event,x,y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.arena = cv2.circle(self.arena, (x, y), 3, 255, -1)
            self.clicked_points.append([x,y])

    def compute_new_average_speed(self, session):
        """
        Calculate the velocity of the mouse. The velocity is the average of the x and y velocities of the body parts.
        Then do |V| = sqrt(Vx^2 + Vy^2) to compute the magnitude of the velocity.
        """
        
        """Here is my attempt of using the direct kalman filter output. However, this is not working well."""
        
        # avgX = np.mean([self.lds_tracking_data[bodypart]['xVelocity'] for bodypart in self.tracking_data_body_parts['bodyparts']], axis=0)
        # avgY = np.mean([self.lds_tracking_data[bodypart]['yVelocity'] for bodypart in self.tracking_data_body_parts['bodyparts']], axis=0)
        # data = np.array([[x, y] for x, y in zip(avgX, avgY)])
        # pixelSpeed = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
        
        # # I think this produces pixesls speed pixels per frame.
        # # not smoothing  because of kalman
        # print()
        # self.region_tracking_data['avg_Velocity'] = (pixelSpeed / session.video.pixels_per_cm)
        # is this in seconds though?
        # self.region_tracking_data['avg_Velocity'] = pixelSpeed * session.video.fps / session.video.pixels_per_cm
        
        """Philips old working code"""
        # Here is the speed of the mouse using the average of the body parts, but not the direct kalman filter output
        # THis is philips old logic but works well
        # Still uses kalman filter positioning
        from scipy.ndimage import gaussian_filter1d
        speed_x_and_y_pixel_per_frame = np.diff(self.region_tracking_data['avg_loc'], axis=0) 
        speed_pixel_per_frame = (speed_x_and_y_pixel_per_frame[:, 0]**2 + speed_x_and_y_pixel_per_frame[:, 1]**2)**.5
        speed_cm_per_sec = speed_pixel_per_frame * session.video.fps / session.video.pixels_per_cm
        self.region_tracking_data['avg_Velocity'] = gaussian_filter1d(speed_cm_per_sec, sigma=session.video.fps/10)
        
    # There seems to be a second component to the old function for the speed calculatuion that is not being used. Leaving as don't understand what it is doing yet.
    # What is the refernece component? 
    
    def compute_speed(self, session, reference_location: tuple = None, reference_name: str=''):
        if not reference_location:
            speed_x_and_y_pixel_per_frame = np.diff(self.tracking_data['avg_loc'], axis=0) 
            speed_pixel_per_frame = (speed_x_and_y_pixel_per_frame[:, 0]**2 + speed_x_and_y_pixel_per_frame[:, 1]**2)**.5
        else:
            distance_from_reference_location = ((self.tracking_data['avg_loc'][:,0] - reference_location[0])**2 + \
                                                (self.tracking_data['avg_loc'][:,1] - reference_location[1])**2)**.5
            self.tracking_data['distance' + reference_name] = distance_from_reference_location
            speed_pixel_per_frame = -np.diff(distance_from_reference_location)
        speed_cm_per_sec = speed_pixel_per_frame * session.video.fps / session.video.pixels_per_cm
        smoothed_speed_cm_per_sec = gaussian_filter1d(speed_cm_per_sec, sigma=session.video.fps/10)
        self.tracking_data['speed' + reference_name] = smoothed_speed_cm_per_sec