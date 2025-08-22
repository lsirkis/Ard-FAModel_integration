from pathlib import Path

import numpy as np

import openmdao.api as om

import ard
from ard.geographic.geomorphology import BathymetryGridData

from famodel import Project
from famodel.platform.platform import Platform
from famodel.helpers import getMoorings, getAnchors, adjustMooring, configureAdjuster

import yaml

from ard.utils.io import load_yaml


class DetailedMooringDesign(om.ExplicitComponent):
    """
    A class to create a detailed mooring design for a floating offshore wind farm.

    This is a class that should be used to generate a floating offshore wind
    farm's collective mooring system.

    Options
    -------
    modeling_options : dict
        a modeling options dictionary (inherited from `FarmAeroTemplate`)
    wind_query : floris.wind_data.WindRose
        a WindQuery objects that specifies the wind conditions that are to be
        computed
    bathymetry_data : ard.geographic.BathymetryData
        a BathymetryData object to specify the bathymetry mesh/sampling

    Inputs
    ------
    phi_platform : np.ndarray
        a 1D numpy array indicating the cardinal direction angle of the mooring
        orientation, with length `N_turbines`
    x_turbines : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    y_turbines : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the turbines,
        with length `N_turbines` (mirrored w.r.t. `FarmAeroTemplate`)
    thrust_turbines : np.ndarray
        an array of the wind turbine thrust for each of the turbines in the farm
        across all of the conditions that have been queried on the wind rose
        (`N_turbines`, `N_wind_conditions`)

    Outputs
    -------
    x_anchors : np.ndarray
        a 1D numpy array indicating the x-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`
    y_anchors : np.ndarray
        a 1D numpy array indicating the y-dimension locations of the mooring
        system anchors, with shape `N_turbines` x `N_anchors`

    """

    def initialize(self):
        """Initialization of the OpenMDAO component."""
        self.options.declare("modeling_options")

        # farm power wind conditions query (not necessarily a full wind rose)
        self.options.declare("wind_query")

        # currently I'm thinking of sea bed conditions as a class, see above
        self.options.declare("bathymetry_data")  # BatyhmetryData object

        self.options.declare("data_path")

    def setup(self):
        """Setup of the OpenMDAO component."""

        # load modeling options
        self.modeling_options = self.options["modeling_options"]
        self.N_turbines = self.modeling_options["farm"]["N_turbines"]
        self.N_anchors = self.modeling_options["platform"]["N_anchors"]
        self.min_mooring_line_length_m = self.modeling_options["platform"][
            "min_mooring_line_length_m"
        ]

        # get the number of wind conditions (for thrust measurements)
        if self.options["wind_query"] is not None:
            self.N_wind_conditions = self.options["wind_query"].N_conditions
        # MANAGE ADDITIONAL LATENT VARIABLES HERE!!!!!

        # BEGIN: VARIABLES TO BE INCORPORATED PROPERLY

        class Placeholder:
            pass  # DEBUG!!!!!

        self.temporary_variables = Placeholder()  # DEBUG!!!!!
        self.temporary_variables.phi_mooring = np.zeros(
            (self.N_turbines,)
        )  # the mooring headings

        # self.temporary_variables.path_to_bathy_moorpy = (
        #     self.options["modeling_options"]["mooring_setup"]["site_conds"]["bathymetry"]["file"]
        # )
        # self.temporary_variables.bathymetry_data = BathymetryGridData()
        # self.temporary_variables.bathymetry_data.load_moorpy_bathymetry(
        #     self.temporary_variables.path_to_bathy_moorpy
        # )
        self.temporary_variables.soil_data = None  # TODO
        self.temporary_variables.radius_fairlead = (
            0.5  # m? idk, replace with a good value
        )
        self.temporary_variables.depth_fairlead = (
            5.0  # m? idk, replace with a good value
        )
        self.temporary_variables.type_anchor = "driven_pile"  # random choice
        # load anchor geometry yaml file based on ard package location
        # self.temporary_variables.path_to_anchor_yaml = (
        #     Path(ard.__file__).parent
        #     / "examples"
        #     / "data"
        #     / "offshore"
        #     / "geometry_anchor.yaml"
        # )
        self.temporary_variables.id_mooring_system = [
            f"m{v}:03d" for v in list(range(len(self.temporary_variables.phi_mooring)))
        ]  # just borrow turbine IDs for now: 3-digit, zero padded integer prefixed by m

        # END VARIABLES TO BE INCORPORATED PROPERLY

        if "mooring_setup" not in self.options["modeling_options"]:
            raise ValueError("Mooring setup options not provided")
        site_depth = self.options["modeling_options"]["site_depth"]
        if "general" in self.options["modeling_options"]["mooring_setup"]["site_conds"]:
            if (
                "water_depth"
                in self.options["modeling_options"]["mooring_setup"]["site_conds"][
                    "general"
                ]
            ):
                water_depth_trow_away = self.options["modeling_options"][
                    "mooring_setup"
                ]["site_conds"]["general"]["water_depth"]
                raise (
                    ValueError(
                        f"'water_depth' ({water_depth_trow_away}) included in 'mooring_setup/site_conds/general', set water depth using 'site_depth'"
                    )
                )

        if (
            "general"
            not in self.options["modeling_options"]["mooring_setup"]["site_conds"]
        ):
            self.options["modeling_options"]["mooring_setup"]["site_conds"][
                "general"
            ] = {}

        # self.options["modeling_options"]["mooring_setup"]["site_conds"]["general"][
        #     "water_depth"
        # ] = site_depth

        if (
            "bathymetry"
            in self.options["modeling_options"]["mooring_setup"]["site_conds"]
        ):
            if (
                "file"
                in self.options["modeling_options"]["mooring_setup"]["site_conds"][
                    "bathymetry"
                ]
            ):
                filename = self.options["modeling_options"]["mooring_setup"][
                    "site_conds"
                ]["bathymetry"]["file"]
                absolute_filename = str(
                    Path(self.options["data_path"]).absolute() / filename
                )
                self.options["modeling_options"]["mooring_setup"]["site_conds"][
                    "bathymetry"
                ]["file"] = absolute_filename

        if "adjuster_settings" in self.options["modeling_options"]["mooring_setup"]:
            if (
                "adjuster"
                in self.options["modeling_options"]["mooring_setup"][
                    "adjuster_settings"
                ]
            ):
                adjuster = self.options["modeling_options"]["mooring_setup"][
                    "adjuster_settings"
                ]["adjuster"]
                if adjuster == "adjustMooring":
                    self.options["modeling_options"]["mooring_setup"][
                        "adjuster_settings"
                    ]["adjuster"] = adjustMooring
        self.FAM = self.buildFAModel(
            **self.options["modeling_options"]["mooring_setup"],
            site_depth=site_depth
        )  # famodel object

        # set up inputs and outputs for mooring system
        self.add_input(
            "phi_platform", np.zeros((self.N_turbines,)), units="deg"
        )  # cardinal direction of the mooring platform orientation
        self.add_input(
            "x_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_input(
            "y_turbines", np.zeros((self.N_turbines,)), units="km"
        )  # y location of the mooring platform in km w.r.t. reference coordinates
        if self.options["wind_query"] is not None:
            self.add_input(
                "thrust_turbines",
                np.zeros((self.N_turbines, self.N_wind_conditions)),
                units="kN",
            )  # turbine thrust coming from each wind direction
        # ADD ADDITIONAL (DESIGN VARIABLE) INPUTS HERE!!!!!

        self.add_output(
            "x_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # x location of the mooring platform in km w.r.t. reference coordinates
        self.add_output(
            "y_anchors",
            np.zeros((self.N_turbines, self.N_anchors)),
            units="km",
        )  # y location of the mooring platform in km w.r.t. reference coordinates

    def setup_partials(self):
        """Derivative setup for the OpenMDAO component."""
        # the default (but not preferred!) derivatives are FDM
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """Computation for the OpenMDAO component."""

        # unpack the working variables
        phi_platform = inputs["phi_platform"]
        x_turbines = inputs["x_turbines"] * 1000
        y_turbines = inputs["y_turbines"] * 1000
        # thrust_turbines = inputs["thrust_turbines"]  # future-proofing

        # BEGIN: ALIASES FOR SOME USEFUL VARIABLES

        phi_mooring = self.temporary_variables.phi_mooring
        # path_to_bathy_moorpy = self.temporary_variables.path_to_bathy_moorpy
        # bathymetry_data = self.temporary_variables.bathymetry_data
        soil_data = self.temporary_variables.soil_data
        radius_fairlead = self.temporary_variables.radius_fairlead
        depth_fairlead = self.temporary_variables.depth_fairlead
        type_anchor = self.temporary_variables.type_anchor
        # path_to_anchor_yaml = self.temporary_variables.path_to_anchor_yaml
        id_mooring_system = self.temporary_variables.id_mooring_system

        # END ALIASES FOR SOME USEFUL VARIABLES

        # reposition FAModel using the x and y turbine postions, and turbine headings
        self.FAM.repositionArray(
            np.array([[x_turbines[i], y_turbines[i]] for i in range(len(x_turbines))]),
            platform_headings=phi_platform,
            anch_resize=False,
            return_costs=True,
        )

        # store anchor x and y positions (km) in lists
        x_anchors = [
            float(self.FAM.anchorList[anch].r[0] / 1000) for anch in self.FAM.anchorList
        ]
        y_anchors = [
            float(self.FAM.anchorList[anch].r[1] / 1000) for anch in self.FAM.anchorList
        ]

        # replace the below with the final anchor locations...
        outputs["x_anchors"] = x_anchors
        outputs["y_anchors"] = y_anchors

    def buildFAModel(self, **FAM_settings):

        # pull out needed information
        pf_coords = FAM_settings.get("pf_locs", np.zeros((self.N_turbines, 2))) # [m]
        pf_headings = FAM_settings.get("pf_headings", np.zeros(self.N_turbines)) # [deg]
        hydrostatics = FAM_settings.get("hydrostatics", {}) # optional dictionary
        RAFT_platform = FAM_settings.get("RAFT_platform", {}) # optional dictionary
        pf_rFair = FAM_settings.get("rFair", 58) # [m]
        pf_zFair = FAM_settings.get("zFair", -14) # [m]
        depth = FAM_settings.get("site_depth", 200) # [m]

        data_path = self.options["data_path"]
        mooring_info = load_yaml(
            Path(data_path).absolute() / FAM_settings["mooring_info"]
        )

        anchor_info = FAM_settings.get("anchor_info", {})
        site_conds = FAM_settings.get("site_conds", {})

        # initialize FAModel project object
        FAM = Project(raft=False, depth=depth)

        # - - - - Site conditions - - - -
        FAM.loadSite(site_conds)

        # - - - - Platforms - - - -
        for i in range(self.N_turbines):

            r = [pf_coords[i][0], pf_coords[i][1], 0]

            if isinstance(pf_rFair, list) or isinstance(pf_rFair, np.ndarray):
                rFair = pf_rFair[i]
            else:
                rFair = pf_rFair
            if isinstance(pf_zFair, list) or isinstance(pf_zFair, np.ndarray):
                zFair = pf_zFair[i]
            else:
                zFair = pf_zFair

            # determine mooring headings and where they are located (needed for platform)
            if "mooring_systems" in mooring_info:
                if len(mooring_info["mooring_systems"]) > 1:
                    raise Exception(
                        "Only one mooring system may be defined for the time being."
                    )
                for m_s in mooring_info["mooring_systems"]:
                    # pull out headings from mooring system
                    # # sort the mooring lines in the mooring system by heading from 0 (North)
                    mySys = [
                        dict(zip(mooring_info["mooring_systems"][m_s]["keys"], row))
                        for row in mooring_info["mooring_systems"][m_s]["data"]
                    ]
                    # get mooring headings (need this for platform class)
                    moor_headings = []
                    for ii in range(0, len(mySys)):
                        moor_headings.append(np.radians(mySys[ii]["heading"]))
            else:
                moor_headings = np.radians(mooring_info["headings"])

            settings = {}
            settings["mooring_headings"] = list(moor_headings)
            if hydrostatics:
                settings["hydrostatics"] = hydrostatics
            elif RAFT_platform:
                settings["raft_platform_dict"] = RAFT_platform

            FAM.addPlatform(
                r=r,
                id=i,
                phi=pf_headings[i],
                entity="FOWT",
                rFair=rFair,
                zFair=zFair,
                **settings,
            )

        # - - - - Anchors - - - -

        lineAnch = None
        count = 0
        for i in range(self.N_turbines):
            for j in range(self.N_anchors):
                if anchor_info:
                    lineAnch = anchor_info
                    atypes = anchor_info
                elif (
                    "anchor_types" in mooring_info and "mooring_systems" in mooring_info
                ):
                    lineAnch = mooring_info["anchor_types"][mySys[j]["anchorType"]]
                    atypes = mooring_info["anchor_types"]
                elif "anchor_types" in mooring_info:
                    anchor_type_name = list(mooring_info["anchor_types"].keys())[0]
                    lineAnch = mooring_info["anchor_types"][anchor_type_name]
                    atypes = mooring_info["anchor_types"]

                FAM.anchorTypes = {}
                for k, v in atypes.items():
                    FAM.anchorTypes[k] = v

                if lineAnch:
                    ad, mass = getAnchors(
                        mySys[j]["anchorType"], arrayAnchor=lineAnch, proj=FAM
                    )  # call method to create anchor dictionary
                else:
                    ad = None  # default
                    mass = 0  # default

                FAM.addAnchor(id=count, dd=ad, mass=mass)
                count += 1

        # - - - - Moorings - - - -
        # make mooring list based on available information

        count = 0
        if "subsystem" in mooring_info:
            for i in range(self.N_turbines):
                for j in range(self.N_anchors):
                    FAM.addMooring(
                        id=count,
                        endA=FAM.anchorList[count],
                        endB=FAM.platformList[i],
                        heading=moor_headings[j] + FAM.platformList[i].phi,
                        subsystem=mooring_info["subsystem"],
                        reposition=True,
                        **FAM_settings["adjuster_settings"],
                    )
                    count += 1

        else:
            lineConfigs = mooring_info["mooring_line_configs"]
            connectorTypes = mooring_info.get("mooring_connector_types", {})

            FAM.lineTypes = {}
            if "mooring_line_types" in mooring_info:
                for k, v in mooring_info["mooring_line_types"].items():
                    # set up line types dictionary
                    FAM.lineTypes[k] = v

            for i in range(self.N_turbines):
                for j in range(self.N_anchors):
                    if "mooring_systems" in mooring_info:
                        lcID = mySys[j]["MooringConfigID"]
                    else:
                        lcID = list(mooring_info["mooring_line_configs"].keys())[0]
                    # create design dictionary of mooring line
                    m_config = getMoorings(
                        lcID,
                        lineConfigs,
                        connectorTypes,
                        pfID=FAM.platformList[i].id,
                        proj=FAM,
                    )

                    # create and attach mooring object
                    FAM.addMooring(
                        id=count,
                        endA=FAM.anchorList[count],
                        endB=FAM.platformList[i],
                        heading=moor_headings[j] + FAM.platformList[i].phi,
                        dd=m_config,
                        reposition=True,
                        **FAM_settings["adjuster_settings"],
                    )
                    count += 1
        
        FAM.updateAnchor()
        FAM.getMoorPyArray()

        return FAM
