# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:09:25 2025

@author: elozon
"""

import pytest
import numpy as np
import openmdao.api as om
from pathlib import Path
import ard
import famodel
from famodel.helpers import adjustMooring
import moorpy as mp
import ruamel.yaml
yaml = ruamel.yaml.YAML()
from copy import deepcopy
from numpy.testing import assert_allclose

class TestMooringDesignDetailed:
    def setup_method(self):

        self.D_rotor = 240.0

        # set turbine layout (3x3 grid 5D spacing)
        X, Y = [
            7.0 * self.D_rotor/1000 * v
            for v in np.meshgrid(np.arange(0, 3), np.arange(0, 3))
        ]

        self.x_turbines = X.flatten()
        self.y_turbines = Y.flatten()

        self.N_turbines = len(self.x_turbines)
        
        self.modeling_options = {
            "farm": {
                "N_turbines": self.N_turbines,
                },
            
            "floating": True,
            "platform": {
                "N_anchors": 3,
                "min_mooring_line_length_m": 500.0,
                "N_anchor_dimensions": 2,
            },
            "site_depth": 50.0,
            "collection": {
                "max_turbines_per_string": 8,
                "solver_name": "appsi_highs",
                "solver_options": dict(
                    time_limit=60,
                    mip_rel_gap=0.005,  # TODO ???
                ),
            },
            "mooring_setup": {
                "site_conds": {
                    "general": {"water_depth": 200},
                    # "bathymetry": {
                    #     "file": Path(ard.__file__).parents[1]
                    #     / "examples"
                    #     / "data"
                    #     / "offshore"
                    #     / "GulfOfMaine_bathymetry_100x99.txt"
                    # },
                    "seabed": {
                        "file": Path(ard.__file__).parents[1]
                        / "examples"
                        / "data"
                        / "offshore"
                        / "GulfOfMaine_soil_100x99.txt"}
                },
                "mooring_info": (
                    Path(ard.__file__).parents[1]
                    / "examples"
                    / "offshore-detailed"
                    / "OntologySample200m.yaml"
                ),
            
                "adjuster_settings": {
                    "adjuster": adjustMooring,
                    "method": "horizontal",
                    "i_line": 1,
                },
            },
        }
               

    def test_FAModel_turbine_positions(self):

        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()

        # set up x and y turbine positions
        prob.set_val("x_turbines", self.x_turbines)
        prob.set_val("y_turbines", self.y_turbines)

        prob.run_model()

        # check that mooring_design turbine positions match the inputs
        assert np.all(
            np.isclose(prob.get_val("mooring_design.x_turbines"), self.x_turbines)
        )
        assert np.all(
            np.isclose(prob.get_val("mooring_design.y_turbines"), self.y_turbines)
        )
   
    def test_FAModel_anchor_positions(self):
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()

        # set up x and y turbine positions to (1 km, 1 km)
        prob.set_val("x_turbines", [1])
        prob.set_val("y_turbines", [1])

        prob.run_model()
        
        # calculate anchor positions in km
        x_anchors = [1 + 0.7*np.cos(60/180*np.pi), 1 + 0.7*np.cos(60/180*np.pi), 1 - 0.7]
        y_anchors = [1 - 0.7*np.sin(60/180*np.pi), 1 + 0.7*np.sin(60/180*np.pi), 1]

        # check that mooring_design anchor positions match expected
        assert np.all(

            np.isclose(prob.get_val("mooring_design.x_anchors"), x_anchors)
        )
        assert np.all(
            np.isclose(prob.get_val("mooring_design.y_anchors"), y_anchors)
        )
    
    def test_FAModel_mooring_config0(self):
        '''
        Check that the mooring configuration matches expected values

        '''
                
        # get mooring yaml
        with open(self.modeling_options["mooring_setup"]["mooring_info"]) as fm:
            msys = dict(yaml.load(fm))
            
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        # pull out a mooring line to check
        m0 = model.mooring_design.FAM.mooringList[0]
        
        # pull out configuration from yaml
        lineprops = mp.helpers.loadLineProps(None)
        yaml_i = 0 # index in yaml sections list
        conf = next(iter(msys['mooring_line_configs'].values()), None) # config
        for sub in m0.subcomponents:
            if yaml_i == len(conf['sections']):
                break
            sec = conf['sections'][yaml_i]
            if 'L' in sub:
                # section
                if 'mooringFamily' in sec:
                    mprop = mp.helpers.getLineProps(sec['d_nom']*1000, 
                                                    sec['mooringFamily'],
                                                    lineProps=lineprops)
                elif 'type' in sec:
                    mprop = msys['mooring_line_types'][sec['type']]
                else:
                    raise Exception('Issue with yaml setup or translation to Mooring')
                # check a few things are the same   
                assert(mprop['d_nom']==sub['type']['d_nom'])
                assert(sec['length']==sub['L'])
                assert(mprop['material']==sub['type']['material'])
                yaml_i += 1
            else:
                # connector
                if sub['m']==0 and sub['v']==0 and sub['CdA']==0:
                    # empty connector, don't add to yaml_i
                    pass
                else:
                    # check a few things are the same
                    assert('connectorType' in sec)
                    assert(sec['connectorType'] in msys['mooring_connector_types'])
                    yaml_i += 1
            
    def test_FAModel_mooring_costs0(self):
        '''
        Check that the cost/m and total cost adds up correctly for unchanged line

        '''
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        # set location of FAModel
        FAM = model.mooring_design.FAM
        # pull out mooring cost of one line
        mCost0 = FAM.mooringList[0].getCost()       
        
        # get mooring yaml
        with open(self.modeling_options["mooring_setup"]["mooring_info"]) as fm:
            msys = dict(yaml.load(fm))
        conf = next(iter(msys['mooring_line_configs'].values()), None) # config
        # get line props for mooring materials
        lineprops = mp.helpers.loadLineProps(None)
        mcosts = []
        for sec in conf['sections']:
            if 'mooringFamily' in sec:
                mprop = mp.helpers.getLineProps(sec['d_nom']*1000, 
                                                sec['mooringFamily'],
                                                lineProps=lineprops)
                mcosts.append(mprop['cost']*sec['length'])
            elif 'type' in sec:
                mprop = msys['mooring_line_types'][sec['type']]
                mcosts.append(mprop['cost']*sec['length'])

            elif 'connectorType' in sec:
                # connector
                # get cost if available
                if sec['connectorType'] in msys['mooring_connector_types']:
                    if 'cost' in msys['mooring_connector_types'][sec['connectorType']]:
                        mcosts.append(msys['mooring_connector_types']
                                      [sec['connectorType']]
                                      ['cost'])                    
        
        # calc costs
        mCost_calc = sum(mcosts)
        assert np.isclose(mCost0, mCost_calc)
                     
                    
    
    def test_FAModel_msystem(self):
        '''
        Check mooring system setup such as number of lines, fairlead radius, fairlead depth, heading
        match expected values
        '''
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        # set location of FAModel
        FAM = model.mooring_design.FAM      
        
        # get mooring yaml
        with open(self.modeling_options["mooring_setup"]["mooring_info"]) as fm:
            msys = dict(yaml.load(fm))
        
        # pull out the first system
        sys = next(iter(msys['mooring_systems'].values()))['data']
        # check number of mooring lines per platform (just 1 pf)
        assert(len(sys)==len(FAM.mooringList))
        sys_heads = [line[1] for line in sys]
        # check rel platform mooring headings match system headings
        assert_allclose(FAM.platformList[0].mooring_headings,np.radians(sys_heads))
        # check mooring abs heading matches system headings (no pf rotation)
        for i,moor in enumerate(FAM.mooringList.values()):
            assert(np.isclose(moor.heading,sys_heads[i]))
            
    def test_FAModel_bathymetry(self):
        ''' Check that bathymetry is loading in to FAModel as expected '''
        # add in varied bathymetry file input
        self.modeling_options["mooring_setup"]["site_conds"]["bathymetry"] = {
                                        "file": Path(ard.__file__).parents[1]
                                        / "examples"
                                        / "data"
                                        / "offshore"
                                        / "GulfOfMaine_bathymetry_100x99.txt"
                                        }
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        
        # set up x and y turbine positions to (1 km, 1 km)
        prob.set_val("x_turbines", [1])
        prob.set_val("y_turbines", [1])

        prob.run_model()
        
        FAM = model.mooring_design.FAM
        
        # first, simply assert there is more than one value in the bath grid/depth
        assert(len(FAM.grid_x)>1)
        assert(len(FAM.grid_depth[0])>1)
        
        # now let's check the actual vals
        bathx, bathy, bathd = famodel.seabed.seabed_tools.readBathymetryFile(
            self.modeling_options["mooring_setup"]["site_conds"]["bathymetry"]["file"])
        assert_allclose(bathx, FAM.grid_x)
        assert_allclose(bathy, FAM.grid_y)
        assert_allclose(bathd, FAM.grid_depth)       
        
        # has to have loaded in to pass above, let's check mooring depth is updated
        m0 = FAM.mooringList[1]
        assert_allclose(m0.rA[2], -FAM.getDepthAtLocation(m0.rA[0],m0.rA[1]))
        a0 = FAM.mooringList[1].attached_to[0] # anchor attached to this mooring
        assert_allclose(m0.rA, a0.r) # check anchor is at same spot
        
    def test_FAModel_soil(self):
        '''Check that soil is loading in to FAModel as expected'''
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        
        # set up x and y turbine positions to (1 km, 1 km)
        prob.set_val("x_turbines", [1])
        prob.set_val("y_turbines", [1])

        prob.run_model()
        
        FAM = model.mooring_design.FAM
        
        # first, simply assert there is more than one value in the soil grids
        assert(len(FAM.soilProps)>0)
        assert(FAM.soil_x is not None)
        assert(FAM.soil_y is not None)
        
        # now let's check the actual vals
        xs, ys, soil_names = famodel.seabed.seabed_tools.readBathymetryFile(
            self.modeling_options["mooring_setup"]["site_conds"]["seabed"]["file"],
            dtype=str)
        sProps = famodel.seabed.seabed_tools.getSoilTypes(
            self.modeling_options["mooring_setup"]["site_conds"]["seabed"]["file"])
        assert_allclose(xs, FAM.soil_x)
        assert_allclose(ys, FAM.soil_y)
        assert soil_names[1,1] == FAM.soil_names[1,1]       
        
        # has to have loaded in nicely to pass above, let's check anchor soil is updated
        a0 = FAM.anchorList[1]
        key, stype = FAM.getSoilAtLocation(a0.r[0],a0.r[1])
        assert({key:stype} == a0.soilProps)
    
    def test_FAModel_mooring_adjust_horizontal(self):
        '''
        Check length adjustment is properly working (test in FAModel checks too,
                                                     but make sure inputs etc haven't changed')
        '''
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # add in varied bathymetry
        self.modeling_options["mooring_setup"]["site_conds"]["bathymetry"] = {
                                        "file": Path(ard.__file__).parents[1]
                                        / "examples"
                                        / "data"
                                        / "offshore"
                                        / "GulfOfMaine_bathymetry_100x99.txt"
                                        }
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        
        # set up x and y turbine positions to (1 km, 1 km)
        prob.set_val("x_turbines", [1])
        prob.set_val("y_turbines", [1])

        prob.run_model()
        
        # set location of FAModel
        FAM = model.mooring_design.FAM 
        
        # get mooring yaml
        with open(self.modeling_options["mooring_setup"]["mooring_info"]) as fm:
            msys = dict(yaml.load(fm))
        # get initial line lengths
        count = 0
        # get index of line to alter length of
        i_line = self.modeling_options["mooring_setup"]["adjuster_settings"]["i_line"]
        for sec in next(iter(msys['mooring_line_configs'].values()),None)['sections']:
            if 'length' in sec:
                if count==i_line:
                    L0=sec['length']
                    break
                count += 1
        
        # first, just check that the lengths are changing with depth for corr line
        for moor in FAM.mooringList.values():
            rA = moor.rA # anchor point
            L = moor.ss.lineList[i_line].L
            if rA[2] != -FAM.depth:
                assert(L != L0)
                if rA[2] > -FAM.depth:
                    # shallower, line should be shorter
                    assert(L < L0)
                elif rA[2] < -FAM.depth:
                    # deeper, line should be longer
                    assert(L > L0)
                    
            # next, check horizontal or total tension is (relatively) constant
            # pull out tension of subsystem, compare to target to be witin 1%
            assert np.isclose(np.linalg.norm(moor.ss.fB_L[:2]), 
                              moor.target, 
                              rtol=.01)

    def test_FAModel_mooring_adjust_pretension(self):
        '''
        Check length adjustment is properly working (test in FAModel checks too,
                                                     but make sure inputs etc haven't changed')
        '''
        #change number of turbines to one
        self.modeling_options["farm"]["N_turbines"] = 1
        
        # revert to constant bath
        if "bathymetry" in self.modeling_options["mooring_setup"]["site_conds"]:
            self.modeling_options["mooring_setup"]["site_conds"].pop("bathymetry")
        # increase depth so we can use taut
        self.modeling_options["mooring_setup"]["site_conds"]["general"]["water_depth"]=800
        # need to switch out the ontology sample too to get a taut line design
        self.modeling_options["mooring_setup"]["mooring_info"] = (
                                        Path(ard.__file__).parents[1]
                                        / "examples"
                                        / "offshore-detailed"
                                        / "OntologySample600m.yaml")
        # switch to pretension adjustment settings
        self.modeling_options["mooring_setup"]["adjuster_settings"] = {
                                                "adjuster": adjustMooring,
                                                "method": "pretension",
                                                "i_line": 1}
        # get anchor positions for flat case
        model0 = om.Group()
        model0.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob0 = om.Problem(model0)
        prob0.setup()
        # set up x and y turbine positions to (1 km, 1 km)
        prob0.set_val("x_turbines", [1])
        prob0.set_val("y_turbines", [1])

        prob0.run_model()
        FAM0 = model0.mooring_design.FAM
        
        # add in varied bathymetry
        self.modeling_options["mooring_setup"]["site_conds"]["bathymetry"] = {
                                        "file": Path(ard.__file__).parents[1]
                                        / "examples"
                                        / "data"
                                        / "offshore"
                                        / "humboldt_bathymetry_100x100.txt"
                                        }
        
        # set up openmdao problem
        model = om.Group()
        model.add_subsystem(  # mooring system design
            "mooring_design",
            ard.offshore.mooring_design_detailed.DetailedMooringDesign(
                modeling_options=self.modeling_options,
                wind_query=None,
            ),
            promotes_inputs=["x_turbines", "y_turbines"],
        )
        
        prob = om.Problem(model)
        prob.setup()
        prob.set_val("x_turbines", [1])
        prob.set_val("y_turbines", [1])
        prob.run_model()
        # set location of FAModel
        FAM = model.mooring_design.FAM 
        # get index of line to alter length of
        i_line = self.modeling_options["mooring_setup"]["adjuster_settings"]["i_line"]
        
        # first, just check that the lengths and anchor locs are changing with depth
        for i,moor in enumerate(FAM.mooringList.values()):
            rA = moor.rA # anchor point
            L = moor.ss.lineList[i_line].L
            L0 = FAM0.mooringList[i].ss.lineList[i_line].L
            if rA[2] != -FAM.depth:
                assert(L != L0)
                if rA[2] > -FAM.depth:
                    # shallower, line should be shorter, smaller span
                    assert(L < L0)
                    # span should also be shorter
                    assert(moor.span < FAM0.mooringList[i].span)
                elif rA[2] < -FAM.depth:
                    # deeper, line should be longer, bigger span
                    assert(L > L0)
                    assert(moor.span > FAM0.mooringList[i].span)
                    
                # check anchor xy loc is changing too    
                assert(any([FAM.anchorList[i].r[j] != FAM0.anchorList[i].r[j] for j in range(2)]))
   
            # next, check total tension is (relatively) constant
            # pull out tension of subsystem, compare to target to be witin 1%
            assert np.isclose(np.linalg.norm(moor.ss.fB_L), 
                              moor.target, 
                              rtol=.01)
        
            
    
    
if __name__ == '__main__':

    testobj = TestMooringDesignDetailed()  
    testobj.setup_method()
    testobj.test_FAModel_turbine_positions()
    testobj.test_FAModel_anchor_positions()
    testobj.test_FAModel_mooring_config0()
    testobj.test_FAModel_mooring_costs0()
    testobj.test_FAModel_msystem()
    testobj.test_FAModel_bathymetry()
    testobj.test_FAModel_soil()
    testobj.test_FAModel_mooring_adjust_horizontal()
    testobj.test_FAModel_mooring_adjust_pretension()

