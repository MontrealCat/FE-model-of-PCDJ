# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import numpy as np

def NSC_Eurocode2(f_cm):
    # Compressive
    e_cu = 0.0035
    e_c = np.arange(0,e_cu+1.0e-6, 5.0e-5)
    sigma_c = np.array([0.0 for i in range(e_c.shape[0])])
    e_in = np.array([0.0 for i in range(e_c.shape[0])])
    d_c = np.array([0.0 for i in range(e_c.shape[0])])
    E_cm = 22000.0 * (0.1 * f_cm) ** 0.3
    e_c1 = min(0.7 * f_cm ** 0.31, 2.8) / 1000.0
    e_oc = 0.4 * f_cm / E_cm
    k = 1.05 * E_cm * e_c1 / f_cm
    ## Compressive stress, inelasitc strain, damage parameter
    for i in range(e_c.shape[0]):
        eita = e_c[i] / e_c1
        if e_c[i] < e_oc:
            sigma_c[i] = (k * eita - eita ** 2) / (1.0 + (k - 2.0) * eita) * f_cm
            e_in[i] = 0.0
        else:
            sigma_c[i] = (k * eita - eita ** 2) / (1.0 + (k - 2.0) * eita) * f_cm
            e_in[i] = e_c[i] - sigma_c[i] / E_cm
        if e_c[i] < e_c1:
            d_c[i] = 0.0
        else:
            d_c[i] = (f_cm - sigma_c[i]) / f_cm
        if sigma_c[i] <0 or d_c[i] > 0.99:
            final_c = i+1
            break
    final_c = i+1
    ## Plastic strain
    e_p_c = e_in - d_c / (1.0 - d_c) * e_oc
    
    # Tensile
    f_tm = 0.1 * f_cm
    e_cr = f_tm / E_cm
    e_0 = 10.0 * e_cr
    e_t = np.arange(0.0,e_0 + 1e-6,5.0e-5)
    e_tcr = np.array([0.0 for i in range(e_t.shape[0])])
    sigma_t = np.array([0.0 for i in range(e_t.shape[0])])
    d_t = np.array([0.0 for i in range(e_t.shape[0])])
    ## Tensile stress, inelasitc strain, damage parameter
    for i in range(e_t.shape[0]):
        if e_t[i] < e_cr:
            sigma_t[i] = f_tm
            e_tcr[i] = 0
            d_t[i] = 0.0
        else:
            sigma_t[i] =  - f_tm / (9.0 * e_cr) * e_t[i] + 10.0 / 9.0 * f_tm
            e_tcr[i] = e_t[i] - sigma_t[i] / E_cm
            d_t[i] = (f_tm - sigma_t[i]) / f_tm
        if sigma_t[i] <0 or d_t[i] > 0.99:
            final_t = i+1
            break
    final_t = i+1
    ## Plastic strain
    e_p_t = e_tcr - d_t / (1.0 - d_t) * sigma_t / E_cm
    
    # Treatment to fit data for CDP in Abaqus
    CCH = []
    DC = []
    CTS = []
    DT = []
    start_c = 0
    start_t = 0
    for i in range(final_c):
        if e_in[i] > 1.0e-7:
            start_c = i-1
            break
        elif e_in[i] < 0:
            e_in[i] = 0
            d_c[i] = 0
    for i in range(final_t):
        if e_tcr[i] > 1.0e-7:
            start_t = i-1
            break
        elif e_tcr[i] < 0:
            e_tcr[i] = 0
            d_t[i] = 0
    for i in range(start_c,final_c):
        CCH.append((round(sigma_c[i],5),round(e_in[i],7)))
        DC.append((round(d_c[i],5),round(e_in[i],7)))
    for i in range(start_t,final_t):
        CTS.append((round(sigma_t[i],5),round(e_tcr[i],7)))
        DT.append((round(d_t[i],5),round(e_tcr[i],7)))
    return E_cm, tuple(CCH), tuple(DC), tuple(CTS), tuple(DT)

def NSC_Eurocode2_M1(f_cm, l_eq):
    # Compressive
    e_cu = 0.0035
    e_c = np.arange(0,e_cu+1.0e-6, 5.0e-5)
    sigma_c = np.array([0.0 for i in range(e_c.shape[0])])
    e_in = np.array([0.0 for i in range(e_c.shape[0])])
    d_c = np.array([0.0 for i in range(e_c.shape[0])])
    E_cm = 22000.0 * (0.1 * f_cm) ** 0.3
    e_c1 = min(0.7 * f_cm ** 0.31, 2.8) / 1000.0
    e_oc = 0.4 * f_cm / E_cm
    k = 1.05 * E_cm * e_c1 / f_cm
    
    f_tm = 0.1 * f_cm
    G_f = 0.073 * f_cm ** 0.18
    G_ch = (f_cm / f_tm) ** 2.0 * G_f
    a_c = 2.0 * 2.5 - 1.0 + 2.0 * np.sqrt(2.5 ** 2.0 - 2.5)
    b_c = 0.4 * f_cm * l_eq / G_ch * (1.0 + a_c / 2.0)
    
    ## Compressive stress, inelasitc strain, damage parameter
    for i in range(e_c.shape[0]):
        eita = e_c[i] / e_c1
        if e_c[i] < e_oc:
            eita = e_oc / e_c1
            sigma_c[i] = (k * eita - eita ** 2) / (1.0 + (k - 2.0) * eita) * f_cm
            e_in[i] = 0.0
        else:
            sigma_c[i] = (k * eita - eita ** 2) / (1.0 + (k - 2.0) * eita) * f_cm
            e_in[i] = e_c[i] - sigma_c[i] / E_cm
        if e_c[i] < e_c1:
            d_c[i] = 1.0 - 1.0 / (2.0 + a_c) * (2.0 * (1.0 + a_c) * np.exp(- b_c * e_in[i]) - a_c * np.exp(- 2.0 * b_c * e_in[i]))
        else:
            d_c[i] = 1.0 - 1.0 / (2.0 + a_c) * (2.0 * (1.0 + a_c) * np.exp(- b_c * e_in[i]) - a_c * np.exp(- 2.0 * b_c * e_in[i]))
        if sigma_c[i] <0 or d_c[i] > 0.99:
            final_c = i+1
            break
    final_c = i+1
    ## Plastic strain
    e_p_c = e_in - d_c / (1.0 - d_c) * e_oc
    
    # Tensile
    e_cr = f_tm / E_cm
    e_0 = 10.0 * e_cr
    e_t = np.arange(0.0,e_0 + 1e-6,5.0e-5)
    e_tcr = np.array([0.0 for i in range(e_t.shape[0])])
    sigma_t = np.array([0.0 for i in range(e_t.shape[0])])
    d_t = np.array([0.0 for i in range(e_t.shape[0])])
    
    a_t = 1
    b_t = f_tm * l_eq / G_f * (1 + a_t / 2)
    
    ## Tensile stress, inelasitc strain, damage parameter
    for i in range(e_t.shape[0]):
        if e_t[i] < e_cr:
            sigma_t[i] = f_tm
            e_tcr[i] = 0
            d_t[i] = 1.0 - 1.0 / (2.0 + a_t) * (2.0 * (1.0 + a_t) * np.exp(- b_t * e_tcr[i]) - a_t * np.exp(- 2.0 * b_t * e_tcr[i]))
        else:
            sigma_t[i] =  - f_tm / (9.0 * e_cr) * e_t[i] + 10.0 / 9.0 * f_tm
            e_tcr[i] = e_t[i] - sigma_t[i] / E_cm
            d_t[i] = 1.0 - 1.0 / (2.0 + a_t) * (2.0 * (1.0 + a_t) * np.exp(- b_t * e_tcr[i]) - a_t * np.exp(- 2.0 * b_t * e_tcr[i]))
        if sigma_t[i] <0 or d_t[i] > 0.99:
            final_t = i+1
            break
    final_t = i+1
    ## Plastic strain
    e_p_t = e_tcr - d_t / (1.0 - d_t) * sigma_t / E_cm
    
    # Treatment to fit data for CDP in Abaqus
    CCH = []
    DC = []
    CTS = []
    DT = []
    start_c = 0
    start_t = 0
    for i in range(final_c):
        if e_in[i] > 1.0e-7:
            start_c = i-1
            break
        elif e_in[i] < 0:
            e_in[i] = 0
            d_c[i] = 0
    for i in range(final_t):
        if e_tcr[i] > 1.0e-7:
            start_t = i-1
            break
        elif e_tcr[i] < 0:
            e_tcr[i] = 0
            d_t[i] = 0
    for i in range(start_c,final_c):
        CCH.append((round(sigma_c[i],5),round(e_in[i],7)))
        DC.append((round(d_c[i],5),round(e_in[i],7)))
    for i in range(start_t,final_t):
        CTS.append((round(sigma_t[i],5),round(e_tcr[i],7)))
        DT.append((round(d_t[i],5),round(e_tcr[i],7)))
    return E_cm, tuple(CCH), tuple(DC), tuple(CTS), tuple(DT)


def DK_model(label, f_c, cs, numk, w_j, d_j, b_j, d_g, w_k, d_k, d_f, d_t, s_k, d_s, Rdia, ept, mat, size0, size1, size2, size3, l_eq):
    # Choice
    w_lp = 100.0   # load_plate width
    h_lp = 20.0    # load_plate height
    h_cp = 20.0    # confining_plate height
    c_c = 20.0
    fri = 0.66
    eccentric = 0.2
    ftf = 1.16
    k_shape = 0.667
    numk = int(numk)
    d_k1 = d_k * numk + (s_k - d_k) * (numk - 1)
    viscosity = 0.0005
    ang = 36.0
    sol = 'I'
    model_name = '{0}-F{1}-E{2}-K{3}-C{4}-M{5}-L{6}-E02'.format(int(label), int(f_c*10), int(ept), int(numk), int(cs), mat, int(l_eq))
    mdb.Model(name=model_name, modelType=STANDARD_EXPLICIT)
    
    ## Epoxy part
    if ept != 0:
        s = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
            sheetSize=1000.0)
        s = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
            sheetSize=1000.0)
        g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
        s.setPrimaryObject(option=STANDALONE)
        s.Line(point1=(0.0, d_j), point2=(0.0, d_j-d_s))
        if numk != 0:
            for i in range(numk):
                s.Line(point1=(0.0, d_j-d_s-s_k*i), point2=(-w_k, d_j-d_s-s_k*i-(d_k-d_f)/2))
                s.Line(point1=(-w_k, d_j-d_s-s_k*i-(d_k-d_f)/2), point2=(-w_k, d_j-d_s-s_k*i-(d_k+d_f)/2))
                s.Line(point1=(-w_k, d_j-d_s-s_k*i-(d_k+d_f)/2), point2=(0.0, d_j-d_s-s_k*i-d_k))
                if i == (numk - 1):
                    s.Line(point1=(0.0, d_j-d_s-s_k*i-d_k), point2=(0.0, 0.0))
                else:
                    s.Line(point1=(0.0, d_j-d_s-s_k*i-d_k), point2=(0.0, d_j-d_s-s_k*(i+1)))
            for i in range(numk):
                s.Line(point1=(0.0-ept, d_j-d_s-s_k*i), point2=(-w_k-ept, d_j-d_s-s_k*i-(d_k-d_f)/2))
                s.Line(point1=(-w_k-ept, d_j-d_s-s_k*i-(d_k-d_f)/2), point2=(-w_k-ept, d_j-d_s-s_k*i-(d_k+d_f)/2))
                s.Line(point1=(-w_k-ept, d_j-d_s-s_k*i-(d_k+d_f)/2), point2=(0.0-ept, d_j-d_s-s_k*i-d_k))
                if i == (numk - 1):
                    s.Line(point1=(0.0-ept, d_j-d_s-s_k*i-d_k), point2=(0.0-ept, 0.0))
                else:
                    s.Line(point1=(0.0-ept, d_j-d_s-s_k*i-d_k), point2=(0.0-ept, d_j-d_s-s_k*(i+1)))
        s.Line(point1=(0.0-ept, d_j-d_s), point2=(0.0-ept, d_j))
        s.Line(point1=(0.0-ept, d_j), point2=(0.0, d_j))
        s.Line(point1=(0.0-ept, 0.0), point2=(0.0, 0.0))
        p = mdb.models[model_name].Part(name='Epoxy', dimensionality=THREE_D, 
            type=DEFORMABLE_BODY)
        p = mdb.models[model_name].parts['Epoxy']
        p.BaseSolidExtrude(sketch=s, depth=b_j)
        s.unsetPrimaryObject()
        p = mdb.models[model_name].parts['Epoxy']
        session.viewports['Viewport: 1'].setValues(displayedObject=p)
        del mdb.models[model_name].sketches['__profile__']
    
    ## Male part
    s = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(0.0, 0.0), point2=(w_j/2, 0.0))
    s.HorizontalConstraint(entity=g[2], addUndoState=False)
    s.Line(point1=(w_j/2, 0.0), point2=(w_j/2, d_j+d_g+d_t))
    s.VerticalConstraint(entity=g[3], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s.Line(point1=(w_j/2, d_j+d_g+d_t), point2=(-w_j/2, d_j+d_g+d_t))
    s.HorizontalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(-w_j/2, d_j+d_g+d_t), point2=(-w_j/2, d_j+d_g))
    s.VerticalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s.Line(point1=(-w_j/2, d_j+d_g), point2=(0.0, d_j+d_g))
    s.HorizontalConstraint(entity=g[6], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    s.Line(point1=(0.0, d_j+d_g), point2=(0.0, d_j))
    s.VerticalConstraint(entity=g[7], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[6], entity2=g[7], addUndoState=False)
    s.Line(point1=(0.0, d_j), point2=(0.0, d_j-d_s))
    s.VerticalConstraint(entity=g[8], addUndoState=False)
    if numk != 0:
        for i in range(numk):
            s.Line(point1=(0.0, d_j-d_s-s_k*i), point2=(-w_k, d_j-d_s-s_k*i-(d_k-d_f)/2))
            s.Line(point1=(-w_k, d_j-d_s-s_k*i-(d_k-d_f)/2), point2=(-w_k, d_j-d_s-s_k*i-(d_k+d_f)/2))
            s.VerticalConstraint(entity=g[10+2*i], addUndoState=False)
            s.Line(point1=(-w_k, d_j-d_s-s_k*i-(d_k+d_f)/2), point2=(0.0, d_j-d_s-s_k*i-d_k))
            if i == (numk - 1):
                s.Line(point1=(0.0, d_j-d_s-s_k*i-d_k), point2=(0.0, 0.0))
            else:
                s.Line(point1=(0.0, d_j-d_s-s_k*i-d_k), point2=(0.0, d_j-d_s-s_k*(i+1)))
            s.VerticalConstraint(entity=g[12+2*i], addUndoState=False)
    p = mdb.models[model_name].Part(name='Male', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['Male']
    p.BaseSolidExtrude(sketch=s, depth=b_j)
    s.unsetPrimaryObject()
    p = mdb.models[model_name].parts['Male']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Female part
    s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.Line(point1=(0.0, 0.0), point2=(0.0, -d_g))
    s1.VerticalConstraint(entity=g[2], addUndoState=False)
    s1.Line(point1=(0.0, -d_g), point2=(w_j/2, -d_g))
    s1.HorizontalConstraint(entity=g[3], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s1.Line(point1=(w_j/2, -d_g), point2=(w_j/2, -(d_t+d_g)))
    s1.VerticalConstraint(entity=g[4], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s1.Line(point1=(w_j/2, -(d_t+d_g)), point2=(-w_j/2, -(d_t+d_g)))
    s1.HorizontalConstraint(entity=g[5], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s1.Line(point1=(-w_j/2, -(d_t+d_g)), point2=(-w_j/2, d_j))
    s1.VerticalConstraint(entity=g[6], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    s1.Line(point1=(-w_j/2, d_j), point2=(0.0, d_j))
    s1.HorizontalConstraint(entity=g[7], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[6], entity2=g[7], addUndoState=False)
    s1.Line(point1=(0.0, d_j), point2=(0.0, d_j-d_s))
    s1.VerticalConstraint(entity=g[8], addUndoState=False)
    if numk != 0:
        for i in range(numk):
            s1.Line(point1=(0.0, d_j-d_s-s_k*i), point2=(-w_k, d_j-d_s-s_k*i-(d_k-d_f)/2))
            s1.Line(point1=(-w_k, d_j-d_s-s_k*i-(d_k-d_f)/2), point2=(-w_k, d_j-d_s-s_k*i-(d_k+d_f)/2))
            s1.VerticalConstraint(entity=g[10+2*i], addUndoState=False)
            s1.Line(point1=(-w_k, d_j-d_s-s_k*i-(d_k+d_f)/2), point2=(0.0, d_j-d_s-s_k*i-d_k))
            if i == (numk - 1):
                s1.Line(point1=(0.0, d_j-d_s-s_k*i-d_k), point2=(0.0, 0.0))
            else:
                s1.Line(point1=(0.0, d_j-d_s-s_k*i-d_k), point2=(0.0, d_j-d_s-s_k*(i+1)))
            s1.VerticalConstraint(entity=g[12+2*i], addUndoState=False)
    p = mdb.models[model_name].Part(name='Female', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['Female']
    p.BaseSolidExtrude(sketch=s1, depth=b_j)
    s1.unsetPrimaryObject()
    p = mdb.models[model_name].parts['Female']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Rebar 1 for Male part
    s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.Line(point1=(w_j/2-c_c, c_c), point2=(w_j/2-c_c, (d_j+d_g+d_t)/2))
    s1.VerticalConstraint(entity=g[2], addUndoState=False)
    s1.Line(point1=(w_j/2-c_c, (d_j+d_g+d_t)/2), point2=(w_j/2-c_c, d_j+d_g+d_t-c_c))
    s1.VerticalConstraint(entity=g[3], addUndoState=False)
    s1.ParallelConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s1.Line(point1=(w_j/2-c_c, d_j+d_g+d_t-c_c), point2=(0.0, d_j+d_g+d_t-c_c))
    s1.HorizontalConstraint(entity=g[4], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s1.Line(point1=(0.0, d_j+d_g+d_t-c_c), point2=(-(w_j/2-c_c), d_j+d_g+d_t-c_c))
    s1.HorizontalConstraint(entity=g[5], addUndoState=False)
    s1.ParallelConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s1.Line(point1=(0.0, d_j+d_g+d_t-c_c), point2=(w_j/2-c_c, (d_j+d_g+d_t)/2))
    p = mdb.models[model_name].Part(name='R1', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['R1']
    p.BaseWire(sketch=s1)
    s1.unsetPrimaryObject()
    p = mdb.models[model_name].parts['R1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Rebar 2 for Female part
    s = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(-(w_j/2-c_c), d_j-c_c), point2=(-(w_j/2-c_c), (d_j-d_g-d_t)/2)) 
    s.VerticalConstraint(entity=g[2], addUndoState=False)
    s.Line(point1=(-(w_j/2-c_c), (d_j-d_g-d_t)/2), point2=(-(w_j/2-c_c), -(d_g+d_t-c_c)))
    s.VerticalConstraint(entity=g[3], addUndoState=False)
    s.ParallelConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s.Line(point1=(-(w_j/2-c_c), -(d_g+d_t-c_c)), point2=(0.0, -(d_g+d_t-c_c)))
    s.HorizontalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(0.0, -(d_g+d_t-c_c)), point2=(w_j/2-c_c, -(d_g+d_t-c_c)))
    s.HorizontalConstraint(entity=g[5], addUndoState=False)
    s.ParallelConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s.Line(point1=(-(w_j/2-c_c), (d_j-d_g-d_t)/2), point2=(0.0, -(d_g+d_t-c_c)))
    p = mdb.models[model_name].Part(name='R2', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['R2']
    p.BaseWire(sketch=s)
    s.unsetPrimaryObject()
    p = mdb.models[model_name].parts['R2']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Rebar 3 for both Male and Female part
    s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.Line(point1=(0.0, 0.0), point2=(b_j-2*c_c, 0.0))
    s1.HorizontalConstraint(entity=g[2], addUndoState=False)
    p = mdb.models[model_name].Part(name='R3', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['R3']
    p.BaseWire(sketch=s1)
    s1.unsetPrimaryObject()
    p = mdb.models[model_name].parts['R3']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Construct MaleRebar with merge function
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    a1 = mdb.models[model_name].rootAssembly
    p = mdb.models[model_name].parts['R1']
    a1.Instance(name='R1-1', part=p, dependent=ON)
    a1 = mdb.models[model_name].rootAssembly
    a1.translate(instanceList=('R1-1', ), vector=(0.0, 0.0, c_c))
    a1 = mdb.models[model_name].rootAssembly
    p = mdb.models[model_name].parts['R1']
    a1.Instance(name='R1-2', part=p, dependent=ON)
    a1 = mdb.models[model_name].rootAssembly
    a1.translate(instanceList=('R1-2', ), vector=(0.0, 0.0, b_j-c_c))
    a1 = mdb.models[model_name].rootAssembly
    p = mdb.models[model_name].parts['R3']
    a1.Instance(name='R3-1', part=p, dependent=ON)
    a1 = mdb.models[model_name].rootAssembly
    a1.rotate(instanceList=('R3-1', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(
        0.0, 1.0, 0.0), angle=90.0)
    a1 = mdb.models[model_name].rootAssembly
    a1.translate(instanceList=('R3-1', ), vector=(w_j/2-c_c, d_j+d_g+d_t-c_c, b_j-c_c))
    a1 = mdb.models[model_name].rootAssembly
    R3num1 = int((d_j+d_g+d_t-2*c_c)//200.0+1)
    R3num2 = int((w_j-2*c_c)//200.0+1)
    a1.LinearInstancePattern(instanceList=('R3-1', ), direction1=(0.0, -1.0, 0.0), 
        direction2=(0.0, 1.0, 0.0), number1=R3num1, number2=1, spacing1=200.0, 
        spacing2=1.0)
    a1 = mdb.models[model_name].rootAssembly
    a1.LinearInstancePattern(instanceList=('R3-1', ), direction1=(-1.0, 0.0, 0.0), 
        direction2=(0.0, 1.0, 0.0), number1=R3num2, number2=1, spacing1=200.0, 
        spacing2=1.0)
    a1 = mdb.models[model_name].rootAssembly
    instance_group = []
    feature_group = []
    instance_group.append(a1.instances['R1-1'])
    instance_group.append(a1.instances['R1-2'])
    instance_group.append(a1.instances['R3-1'])
    feature_group.append('R1-1')
    feature_group.append('R1-2')
    feature_group.append('R3-1')
    for i in range(1, max(R3num1,R3num2)):
        instance_group.append(a1.instances['R3-1-lin-{0}-1'.format(i+1)])
        feature_group.append('R3-1-lin-{0}-1'.format(i+1))
    for i in range(1, min(R3num1,R3num2)):
        instance_group.append(a1.instances['R3-1-lin-{0}-1-1'.format(i+1)])
        feature_group.append('R3-1-lin-{0}-1-1'.format(i+1))
    MaleR_group = tuple(instance_group)
    MaleR_features = tuple(feature_group)
    a1.InstanceFromBooleanMerge(name='MaleRebar', instances=MaleR_group, originalInstances=SUPPRESS, 
        domain=GEOMETRY)
    a = mdb.models[model_name].rootAssembly
    a.deleteFeatures(MaleR_features)

    ## Construct FemaleRebar with merge function
    a1 = mdb.models[model_name].rootAssembly
    p = mdb.models[model_name].parts['R2']
    a1.Instance(name='R2-1', part=p, dependent=ON)
    a1 = mdb.models[model_name].rootAssembly
    a1.translate(instanceList=('R2-1', ), vector=(0.0, 0.0, c_c))
    a1 = mdb.models[model_name].rootAssembly
    p = mdb.models[model_name].parts['R2']
    a1.Instance(name='R2-2', part=p, dependent=ON)
    a1 = mdb.models[model_name].rootAssembly
    a1.translate(instanceList=('R2-2', ), vector=(0.0, 0.0, b_j-c_c))
    a1 = mdb.models[model_name].rootAssembly
    p = mdb.models[model_name].parts['R3']
    a1.Instance(name='R3-1', part=p, dependent=ON)
    a1 = mdb.models[model_name].rootAssembly
    a1.rotate(instanceList=('R3-1', ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(
        0.0, 1.0, 0.0), angle=90.0)
    a1 = mdb.models[model_name].rootAssembly
    a1.translate(instanceList=('R3-1', ), vector=(-(w_j/2-c_c), -(d_g+d_t-c_c), b_j-c_c))
    a1 = mdb.models[model_name].rootAssembly
    a1.LinearInstancePattern(instanceList=('R3-1', ), direction1=(0.0, 1.0, 0.0), 
        direction2=(0.0, 1.0, 0.0), number1=R3num1, number2=1, spacing1=200.0, 
        spacing2=1.0)
    a1 = mdb.models[model_name].rootAssembly
    a1.LinearInstancePattern(instanceList=('R3-1', ), direction1=(1.0, 0.0, 0.0), 
        direction2=(0.0, 1.0, 0.0), number1=R3num2, number2=1, spacing1=200.0, 
        spacing2=1.0)
    a1 = mdb.models[model_name].rootAssembly
    instance_group = []
    feature_group = []
    instance_group.append(a1.instances['R2-1'])
    instance_group.append(a1.instances['R2-2'])
    instance_group.append(a1.instances['R3-1'])
    feature_group.append('R2-1')
    feature_group.append('R2-2')
    feature_group.append('R3-1')
    for i in range(1, max(R3num1,R3num2)):
        instance_group.append(a1.instances['R3-1-lin-{0}-1'.format(i+1)])
        feature_group.append('R3-1-lin-{0}-1'.format(i+1))
    for i in range(1, min(R3num1,R3num2)):
        instance_group.append(a1.instances['R3-1-lin-{0}-1-1'.format(i+1)])
        feature_group.append('R3-1-lin-{0}-1-1'.format(i+1))
    FemaleR_group = tuple(instance_group)
    FemaleR_features = tuple(feature_group)
    a1.InstanceFromBooleanMerge(name='FemaleRebar', instances=FemaleR_group, originalInstances=SUPPRESS, 
        domain=GEOMETRY)
    a = mdb.models[model_name].rootAssembly
    a.deleteFeatures(FemaleR_features)
        
    ## Loading plate
    s = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(-w_lp/2, d_j+d_g+d_t), point2=(w_lp/2, d_j+d_g+d_t))
    s.HorizontalConstraint(entity=g[2], addUndoState=False)
    s.Line(point1=(w_lp/2, d_j+d_g+d_t), point2=(w_lp/2, d_j+d_g+d_t+h_lp))
    s.VerticalConstraint(entity=g[3], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s.Line(point1=(w_lp/2, d_j+d_g+d_t+h_lp), point2=(-w_lp/2, d_j+d_g+d_t+h_lp))
    s.HorizontalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(-w_lp/2, d_j+d_g+d_t+h_lp), point2=(-w_lp/2, d_j+d_g+d_t))
    s.VerticalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    p = mdb.models[model_name].Part(name='LoadPlate', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['LoadPlate']
    p.BaseSolidExtrude(sketch=s, depth=b_j)
    s.unsetPrimaryObject()
    p = mdb.models[model_name].parts['LoadPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Left Confining stress plate
    s1 = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.Line(point1=(-w_j/2, 0.0), point2=(-w_j/2, d_j))
    s1.VerticalConstraint(entity=g[2], addUndoState=False)
    s1.Line(point1=(-w_j/2, d_j), point2=(-(w_j/2+h_cp), d_j))
    s1.HorizontalConstraint(entity=g[3], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s1.Line(point1=(-(w_j/2+h_cp), d_j), point2=(-(w_j/2+h_cp), 0.0))
    s1.VerticalConstraint(entity=g[4], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s1.Line(point1=(-(w_j/2+h_cp), 0.0), point2=(-w_j/2, 0.0))
    s1.HorizontalConstraint(entity=g[5], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    p = mdb.models[model_name].Part(name='CSLPlate', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['CSLPlate']
    p.BaseSolidExtrude(sketch=s1, depth=b_j)
    s1.unsetPrimaryObject()
    p = mdb.models[model_name].parts['CSLPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    ## Right Confining stress plate
    s = mdb.models[model_name].ConstrainedSketch(name='__profile__', 
        sheetSize=1000.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(w_j/2, 0.0), point2=(w_j/2, d_j))
    s.VerticalConstraint(entity=g[2], addUndoState=False)
    s.Line(point1=(w_j/2, d_j), point2=(w_j/2+h_cp, d_j))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s.Line(point1=(w_j/2+h_cp, d_j), point2=(w_j/2+h_cp, 0.0))
    s.VerticalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(w_j/2+h_cp, 0.0), point2=(w_j/2, 0.0))
    s.HorizontalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    p = mdb.models[model_name].Part(name='CSRPlate', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models[model_name].parts['CSRPlate']
    p.BaseSolidExtrude(sketch=s, depth=b_j)
    s.unsetPrimaryObject()
    p = mdb.models[model_name].parts['CSRPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[model_name].sketches['__profile__']

    # Step 2: Generate Mesh
    ## Epoxy
    if ept != 0:
        if numk != 0:
            p = mdb.models[model_name].parts['Epoxy']
            session.viewports['Viewport: 1'].setValues(displayedObject=p)
            p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-ept)
            p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-w_k)
            c = p.cells
            d = p.datums
            pickedCells = c.findAt(((0.0,0.0,0.0),))
            p.PartitionCellByDatumPlane(datumPlane=d[3], cells=pickedCells)
            pickedCells = c.findAt(((0.0,0.0,0.0),), ((0.0,d_j,0.0),))
            p.PartitionCellByDatumPlane(datumPlane=d[2], cells=pickedCells)
            p.seedPart(size=ept, deviationFactor=0.1, minSizeFactor=0.1)
            p.generateMesh()
        else:
            p = mdb.models[model_name].parts['Epoxy']
            p.seedPart(size=ept, deviationFactor=0.1, minSizeFactor=0.1)
            p.generateMesh()
        
    ## CSL plate
    p = mdb.models[model_name].parts['CSLPlate']
    p.seedPart(size=size1, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['CSLPlate']
    p.generateMesh()

    ## CSR plate
    p = mdb.models[model_name].parts['CSRPlate']
    p.seedPart(size=size1, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['CSRPlate']
    p.generateMesh()

    ## Load plate
    p = mdb.models[model_name].parts['LoadPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['LoadPlate']
    p.seedPart(size=size1, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['LoadPlate']
    p.generateMesh()

    ## FemaleRebar
    p = mdb.models[model_name].parts['FemaleRebar']
    p.seedPart(size=size3, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['FemaleRebar']
    p.generateMesh()
    elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=STANDARD)
    p = mdb.models[model_name].parts['FemaleRebar']
    e = p.edges
    edges = e.getByBoundingBox(-w_j/2-1, -d_g-d_t-1, 0-1, w_j/2+1 ,d_j+1 ,b_j+1)
    pickedRegions =(edges, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))

    ## MaleRebar
    p = mdb.models[model_name].parts['MaleRebar']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['MaleRebar']
    p.seedPart(size=size3, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models[model_name].parts['MaleRebar']
    p.generateMesh()
    elemType1 = mesh.ElemType(elemCode=T3D2, elemLibrary=STANDARD)
    p = mdb.models[model_name].parts['MaleRebar']
    e = p.edges
    edges = e.getByBoundingBox(-w_j/2-1, 0-1, 0-1, w_j/2+1 ,d_j+d_g+d_t+1 ,b_j+1)
    pickedRegions =(edges, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))

    ## Female part
    p = mdb.models[model_name].parts['Female']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-w_lp/2)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=w_lp/2)
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=0.0)
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=-d_g)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-w_k)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0.0)
    c = p.cells
    d = p.datums
    # -d_g horizontal
    pickedCells = c.findAt(((-w_j/2,d_j/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[5], cells=pickedCells)
    # 0 horizontal
    pickedCells = c.findAt(((-w_j/2,d_j/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[4], cells=pickedCells)
    if numk != 0:
        # -w_j vertical up
        pickedCells = c.findAt(((-w_j/2,d_j/2,b_j/2),))
        p.PartitionCellByDatumPlane(datumPlane=d[6], cells=pickedCells)
        # -w_j vertical mid
        pickedCells = c.findAt(((-w_j/2,-d_g/2,b_j/2),))
        p.PartitionCellByDatumPlane(datumPlane=d[6], cells=pickedCells)
        # -w_j vertical bottom
        pickedCells = c.findAt(((w_j/2,-d_g-d_t/2,b_j/2),))
        p.PartitionCellByDatumPlane(datumPlane=d[6], cells=pickedCells)
        # w_j vertical
        pickedCells = c.findAt(((w_j/2,-d_g-d_t/2,b_j/2),))
        p.PartitionCellByDatumPlane(datumPlane=d[3], cells=pickedCells)
    # 0 vertical
    pickedCells = c.findAt(((0,-d_g-d_t/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[7], cells=pickedCells)

    p = mdb.models[model_name].parts['Female']
    p.seedPart(size=size0, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, 
        kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)

    p = mdb.models[model_name].parts['Female']
    e = p.edges
    FFaces1 = [((-w_k/2,0.0,b_j),), ((-w_k/2,-d_g,b_j),), ((-w_k/2,d_j,b_j),), 
    ((-w_k,d_j-d_s/2-(d_k-d_f)/4,b_j),),((0.0,d_j-d_s/2,b_j),),((50.0/2,-d_g,b_j),)]
    FFaces2 = []
    for i in range(numk):
        FFaces1.append(((-w_k,d_j-d_s-s_k*i-d_k/2,b_j), ))
        FFaces2.append(((-w_k/2,d_j-d_s-s_k*i-(d_k-d_f)/4,b_j), ))
        FFaces2.append(((-w_k/2,d_j-d_s-s_k*i-(3*d_k+d_f)/4,b_j), ))
        if i == (numk - 1):
            FFaces2.append(((0.0,(d_j-d_s-s_k*i-d_k)/2,b_j), ))
            FFaces1.append(((-w_k,(d_j-d_s-s_k*i-d_k)/2,b_j), ))
        else:
            FFaces2.append(((0.0,d_j-d_s-s_k*i-(d_k+s_k)/2,b_j), ))
            FFaces2.append(((-w_k,d_j-d_s-s_k*i-(d_k+s_k)/2,b_j), ))
    pickedEdges = e.findAt(*FFaces1)
    p.seedEdgeBySize(edges=pickedEdges, size=size1, deviationFactor=0.1, 
        minSizeFactor=0.1, constraint=FINER)
    p = mdb.models[model_name].parts['Female']
    if numk != 0:
        pickedEdges = e.findAt(*FFaces2)
        p.seedEdgeBySize(edges=pickedEdges, size=size2, deviationFactor=0.1, 
            minSizeFactor=0.1, constraint=FINER)     
    p.generateMesh()

    ## Male part
    p = mdb.models[model_name].parts['Male']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-w_lp/2)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=w_lp/2)
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=d_j)
    p.DatumPlaneByPrincipalPlane(principalPlane=XZPLANE, offset=d_j+d_g)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=-w_k)
    p.DatumPlaneByPrincipalPlane(principalPlane=YZPLANE, offset=0.0)
    c = p.cells
    d = p.datums
    # d_j+d_g horizontal
    pickedCells = c.findAt(((w_j/2,d_j/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[5], cells=pickedCells)
    # d_j horizontal
    pickedCells = c.findAt(((w_j/2,d_j/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[4], cells=pickedCells)
    if numk != 0:
        # 0 vertical bottom
        pickedCells = c.findAt(((w_j/2,d_j/2,b_j/2),))
        p.PartitionCellByDatumPlane(datumPlane=d[7], cells=pickedCells)
    # 50 vertical bottom
    pickedCells = c.findAt(((w_j/2,d_j/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=pickedCells)
    # 50 vertical mid
    pickedCells = c.findAt(((w_j/2,d_j+d_g/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=pickedCells)
    # 50 vertical up
    pickedCells = c.findAt(((-w_j/2,d_j+d_g+d_t/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[3], cells=pickedCells)
    # -50 vertical
    pickedCells = c.findAt(((-w_j/2,d_j+d_g+d_t/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[2], cells=pickedCells)
    # 0 vertical up
    pickedCells = c.findAt(((0,d_j+d_g+d_t/2,b_j/2),))
    p.PartitionCellByDatumPlane(datumPlane=d[7], cells=pickedCells)
     
    p = mdb.models[model_name].parts['Male']
    p.seedPart(size=size0, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD, 
        kinematicSplit=AVERAGE_STRAIN, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT)
    elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
    elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)

    p = mdb.models[model_name].parts['Male']
    e = p.edges
    MFaces1 = [((w_lp/4,0.0,b_j),), ((w_lp/4,d_j,b_j),), ((w_lp/4,d_j+d_g,b_j),), ((-w_lp/4,d_j+d_g,b_j),),
    ((w_lp/2,d_j/2,b_j),), ((w_lp/2,d_j+d_g/2,b_j),), ((0.0,d_j+d_g/2,b_j),), ((0.0,d_j-d_s/2,b_j),)]
    MFaces2 = []
    for i in range(numk):
        MFaces2.append(((-w_k/2,d_j-d_s-s_k*i-(d_k-d_f)/4,b_j), ))
        MFaces2.append(((-w_k,d_j-d_s-s_k*i-d_k/2,b_j), ))
        MFaces2.append(((-w_k/2,d_j-d_s-s_k*i-(3*d_k+d_f)/4,b_j), ))
        MFaces2.append(((0.0,d_j-d_s-s_k*i-d_k/2,b_j), ))
        if i == (numk - 1):
            MFaces1.append(((0.0,(d_j-d_s-s_k*i-d_k)/2,b_j), ))
        else:
            MFaces1.append(((0.0,d_j-d_s-s_k*i-(d_k+s_k)/2,b_j), ))
    pickedEdges = e.findAt(*MFaces1)
    p.seedEdgeBySize(edges=pickedEdges, size=size1, deviationFactor=0.1, 
        minSizeFactor=0.1, constraint=FINER)
    p = mdb.models[model_name].parts['Male']
    if numk != 0:
        pickedEdges = e.findAt(*MFaces2)
        p.seedEdgeBySize(edges=pickedEdges, size=size2, deviationFactor=0.1, 
            minSizeFactor=0.1, constraint=FINER)
    p.generateMesh()

    # Step 3: Material
    ## Definition
    if mat == 'E':
        E_cm, CCH, DC, CTS, DT = NSC_Eurocode2(f_c)
    elif mat == 'E1':
        E_cm, CCH, DC, CTS, DT = NSC_Eurocode2_M1(f_c, l_eq)
    elif mat == 'F':
        E_cm, CCH, DC, CTS, DT = NSC_CEB_fib(f_c, l_eq)
    elif mat == 'F1':
        E_cm, CCH, DC, CTS, DT = NSC_CEB_fib_M1(f_c, l_eq)
    elif mat == 'C':
        E_cm, CCH, DC, CTS, DT = NSC_China(f_c, l_eq)
    elif mat == 'C1':
        E_cm, CCH, DC, CTS, DT = NSC_China_M1(f_c, l_eq)    
    mdb.models[model_name].Material(name='NSC')
    mdb.models[model_name].materials['NSC'].Density(table=((2.3e-09, ), ))
    mdb.models[model_name].materials['NSC'].Elastic(table=((E_cm, 0.2), ))
    mdb.models[model_name].materials['NSC'].ConcreteDamagedPlasticity(table=((ang, 
        eccentric, ftf, k_shape, viscosity), ))
    mdb.models[model_name].materials['NSC'].concreteDamagedPlasticity.ConcreteCompressionHardening(
        table=CCH)
    mdb.models[model_name].materials['NSC'].concreteDamagedPlasticity.ConcreteTensionStiffening(
        table=CTS)
    mdb.models[model_name].materials['NSC'].concreteDamagedPlasticity.ConcreteCompressionDamage(
        table=DC)
    mdb.models[model_name].materials['NSC'].concreteDamagedPlasticity.ConcreteTensionDamage(
        table=DT)
    mdb.models[model_name].Material(name='Rebar')
    mdb.models[model_name].materials['Rebar'].Density(table=((7.85e-09, ), ))
    mdb.models[model_name].materials['Rebar'].Elastic(table=((210000.0, 0.3), ))
    mdb.models[model_name].materials['Rebar'].Plastic(table=((400.0, 0.0), (400.0, 
        0.1)))
    mdb.models[model_name].Material(name='Steel')
    mdb.models[model_name].materials['Steel'].Density(table=((7.85e-09, ), ))
    mdb.models[model_name].materials['Steel'].Elastic(table=((210000.0, 0.3), ))

    ## Assign to Geometry
    mdb.models[model_name].HomogeneousSolidSection(name='S-CSLPlate', 
        material='Steel', thickness=None)
    mdb.models[model_name].HomogeneousSolidSection(name='S-CSRPlate', 
        material='Steel', thickness=None)
    mdb.models[model_name].HomogeneousSolidSection(name='S-LoadPlate', 
        material='Steel', thickness=None) 
    mdb.models[model_name].TrussSection(name='S-FemaleRebar', material='Rebar', 
        area=3.1416*Rdia**2/4)
    mdb.models[model_name].TrussSection(name='S-MaleRebar', material='Rebar', 
        area=3.1416*Rdia**2/4)
    mdb.models[model_name].HomogeneousSolidSection(name='S-Female', material='NSC', 
        thickness=None)
    mdb.models[model_name].HomogeneousSolidSection(name='S-Male', material='NSC', 
        thickness=None)

    p = mdb.models[model_name].parts['CSLPlate']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(cells=cells, name='Set-CSLPlate')
    #: The set 'Set-CSLPlate' has been created (1 cell).

    p = mdb.models[model_name].parts['CSRPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['CSRPlate']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(cells=cells, name='Set-CSRPlate')
    #: The set 'Set-CSRPlate' has been created (1 cell).

    p = mdb.models[model_name].parts['LoadPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['LoadPlate']
    c = p.cells
    cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(cells=cells, name='Set-LoadPlate')
    #: The set 'Set-LoadPlate' has been created (1 cell).

    m1 = ['[#ff ]', '[#1ff ]', '[#3ff ]', '[#7ff ]', '[#fff ]', '[#1fff ]', '[#3fff ]', '[#7fff ]', '[#ffff ]',
    '[#1ffff ]', '[#3ffff ]', '[#7ffff ]', '[#fffff ]']
    p = mdb.models[model_name].parts['Female']
    c = p.cells
    cells = c.getSequenceFromMask(mask=(m1[numk], ), )
    p.Set(cells=cells, name='Set-Female')
    # : The set 'Set-Female' has been created 
    p = mdb.models[model_name].parts['FemaleRebar']
    e = p.edges
    edges = e.getByBoundingBox(-w_j/2-1, -d_g-d_t-1, 0-1, w_j/2+1 ,d_j+1 ,b_j+1)
    p.Set(edges=edges, name='Set-FemaleRebar')
    # : The set 'Set-FemaleRebar' has been created 
    p = mdb.models[model_name].parts['Male']
    c = p.cells
    cells = c.getSequenceFromMask(mask=(m1[numk], ), )
    p.Set(cells=cells, name='Set-Male')
    # : The set 'Set-Male' has been created
    p = mdb.models[model_name].parts['MaleRebar']
    e = p.edges
    edges = e.getByBoundingBox(-w_j/2-1, 0-1, 0-1, w_j/2+1 ,d_j+d_g+d_t+1 ,b_j+1)
    p.Set(edges=edges, name='Set-MaleRebar')
    # : The set 'Set-MaleRebar' has been created 
    
    ## Assign
    p = mdb.models[model_name].parts['CSLPlate']
    region = p.sets['Set-CSLPlate']
    p = mdb.models[model_name].parts['CSLPlate']
    p.SectionAssignment(region=region, sectionName='S-CSLPlate', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    p = mdb.models[model_name].parts['CSRPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['CSRPlate']
    region = p.sets['Set-CSRPlate']
    p = mdb.models[model_name].parts['CSRPlate']
    p.SectionAssignment(region=region, sectionName='S-CSRPlate', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    p = mdb.models[model_name].parts['Female']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['Female']
    region = p.sets['Set-Female']
    p = mdb.models[model_name].parts['Female']
    p.SectionAssignment(region=region, sectionName='S-Female', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    p = mdb.models[model_name].parts['FemaleRebar']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['FemaleRebar']
    region = p.sets['Set-FemaleRebar']
    p = mdb.models[model_name].parts['FemaleRebar']
    p.SectionAssignment(region=region, sectionName='S-FemaleRebar', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    p = mdb.models[model_name].parts['LoadPlate']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['LoadPlate']
    region = p.sets['Set-LoadPlate']
    p = mdb.models[model_name].parts['LoadPlate']
    p.SectionAssignment(region=region, sectionName='S-LoadPlate', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    p = mdb.models[model_name].parts['Male']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['Male']
    region = p.sets['Set-Male']
    p = mdb.models[model_name].parts['Male']
    p.SectionAssignment(region=region, sectionName='S-Male', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
        
    p = mdb.models[model_name].parts['MaleRebar']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[model_name].parts['MaleRebar']
    region = p.sets['Set-MaleRebar']
    p = mdb.models[model_name].parts['MaleRebar']
    p.SectionAssignment(region=region, sectionName='S-MaleRebar', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)

    # Step 4: Create Assembly and Contact
    ## Add parts
    a1 = mdb.models[model_name].rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = mdb.models[model_name].parts['CSLPlate']
    a1.Instance(name='CSLPlate-1', part=p, dependent=ON)
    p = mdb.models[model_name].parts['CSRPlate']
    a1.Instance(name='CSRPlate-1', part=p, dependent=ON)
    p = mdb.models[model_name].parts['Female']
    a1.Instance(name='Female-1', part=p, dependent=ON)
    p = mdb.models[model_name].parts['Male']
    a1.Instance(name='Male-1', part=p, dependent=ON)
    p = mdb.models[model_name].parts['LoadPlate']
    a1.Instance(name='LoadPlate-1', part=p, dependent=ON)
    if ept != 0:
        p = mdb.models[model_name].parts['Epoxy']
        a1.Instance(name='Epoxy-1', part=p, dependent=ON)
        a = mdb.models[model_name].rootAssembly
        a.translate(instanceList=('Female-1', 'CSLPlate-1', 'FemaleRebar-1'), vector=(-ept, 0.0, 0.0))
    
    ## Create sets
    a = mdb.models[model_name].rootAssembly
    c1 = a.instances['LoadPlate-1'].cells
    cells1 = c1.findAt(((0,d_j+d_g+d_t+h_lp/2,b_j/2),))
    a.Set(cells=cells1, name='Set-LoadP')
    #: The set 'Set-LP' has been created (1 cell).

    a = mdb.models[model_name].rootAssembly
    c1 = a.instances['CSLPlate-1'].cells
    cells1 = c1.findAt(((-(w_j/2+h_cp/2)-ept,d_j/2,b_j/2),))
    a.Set(cells=cells1, name='Set-CSLP')
    #: The set 'Set-CSLP' has been created (1 cell).

    a = mdb.models[model_name].rootAssembly
    c1 = a.instances['CSRPlate-1'].cells
    cells1 = c1.findAt(((w_j/2+h_cp/2,d_j/2,b_j/2),))
    a.Set(cells=cells1, name='Set-CSR')
    #: The set 'Set-CSRP' has been created (1 cell).

    a = mdb.models[model_name].rootAssembly
    rp1 = a.ReferencePoint(point=(0.0, (d_j+d_t+d_g)+h_lp+60.0, b_j/2))
    a = mdb.models[model_name].rootAssembly
    r1 = a.referencePoints
    refPoints1=(r1[rp1.id], )
    a.Set(referencePoints=refPoints1, name='Set-RP')
    # : The set 'Set-RP' has been created (1 reference point).

    m1 = ['[#ff ]', '[#1ff ]', '[#3ff ]', '[#7ff ]', '[#fff ]', '[#1fff ]', '[#3fff ]', '[#7fff ]', '[#ffff ]',
    '[#1ffff ]', '[#3ffff ]', '[#7ffff ]', '[#fffff ]']
    a = mdb.models[model_name].rootAssembly
    c1 = a.instances['Female-1'].cells
    cells1 = c1.getSequenceFromMask(mask=(m1[numk], ), )
    a.Set(cells=cells1, name='Set-Female')
    #: The set 'Set-Female' has been created (9 cells).
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['FemaleRebar-1'].edges
    edges1 = e1.getByBoundingBox(-w_j/2-1-ept, -d_g-d_t-1, 0-1, w_j/2+1-ept ,d_j+1 ,b_j+1)
    a.Set(edges=edges1, name='Set-FemaleRebar')
    #: The set 'Set-FemaleRebar' has been created (56 edges).
    a = mdb.models[model_name].rootAssembly
    c1 = a.instances['Male-1'].cells
    cells1 = c1.getSequenceFromMask(mask=(m1[numk], ), )
    a.Set(cells=cells1, name='Set-Male')
    #: The set 'Set-Male' has been created (9 cells).
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['MaleRebar-1'].edges
    edges1 = e1.getByBoundingBox(-w_j/2-1, 0-1, 0-1, w_j/2+1 ,d_j+d_g+d_t+1 ,b_j+1)
    a.Set(edges=edges1, name='Set-MaleRebar')
    #: The set 'Set-MaleRebar' has been created (56 edges).
    a = mdb.models[model_name].rootAssembly
    f1 = a.instances['Female-1'].faces
    faces1 = f1.findAt(((-w_k/2-ept,-(d_g+d_t),b_j/2),), ((-w_k-(w_j/2-w_k)/2-ept,-(d_g+d_t),b_j/2),), 
    ((w_lp/4-ept,-(d_g+d_t),b_j/2),), ((w_lp/2+(w_j/2-w_lp/2)/2-ept,-(d_g+d_t),b_j/2),))
    a.Set(faces=faces1, name='Set-Constraint')
    #: The set 'Set-Constraint' has been created (4 faces).

    ## Create surfaces
    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['LoadPlate-1'].faces
    side1Faces1 = s1.findAt(((0,d_j+d_g+d_t,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='LoadP-B')
    #: The surface 'LoadP-B' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['LoadPlate-1'].faces
    side1Faces1 = s1.findAt(((0,d_j+d_g+d_t+h_lp,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='LoadP-T')
    #: The surface 'LoadP-T' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['CSLPlate-1'].faces
    side1Faces1 = s1.findAt(((-w_j/2-ept,d_j/2,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='CSLP-R')
    #: The surface 'CSLP-R' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['CSLPlate-1'].faces
    side1Faces1 = s1.findAt(((-(w_j/2+h_cp)-ept,d_j/2,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='CSLP-L')
    #: The surface 'CSLP-L' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['CSRPlate-1'].faces
    side1Faces1 = s1.findAt(((w_j/2,d_j/2,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='CSRP-L')
    #: The surface 'CSRP-L' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['CSRPlate-1'].faces
    side1Faces1 = s1.findAt(((w_j/2+h_cp,d_j/2,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='CSRP-R')
    #: The surface 'CSRP-R' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['Male-1'].faces
    side1Faces1 = s1.findAt(((w_lp/4,d_j+d_g+d_t,b_j/2),), ((-w_lp/4,d_j+d_g+d_t,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='Male-T')
    #: The surface 'Male-T' has been created (2 faces).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['Male-1'].faces
    side1Faces1 = s1.findAt(((w_j/2,d_j/2,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='Male-R')
    #: The surface 'Male-R' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['Female-1'].faces
    side1Faces1 = s1.findAt(((-w_k/2-ept,-(d_g+d_t),b_j/2),), ((-w_k-(w_j/2-w_k)/2-ept,-(d_g+d_t),b_j/2),), 
    ((w_lp/4-ept,-(d_g+d_t),b_j/2),), ((w_lp/2+(w_j/2-w_lp/2)/2-ept,-(d_g+d_t),b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='Female-B')
    #: The surface 'Female-B' has been created (2 faces).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['Female-1'].faces
    side1Faces1 = s1.findAt(((-w_j/2-ept,d_j/2,b_j/2),))
    a.Surface(side1Faces=side1Faces1, name='Female-L')
    #: The surface 'Female-L' has been created (1 face).

    a = mdb.models[model_name].rootAssembly
    s1 = a.instances['Male-1'].faces
    if numk == 0: 
        Ffaces = [((0,d_j-d_s/2,b_j/2),)]
        if ept != 0:
            Ffaces2 = [((0-ept,d_j-d_s/2,b_j/2),)]
            side1Faces1 = a.instances['Epoxy-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Epoxy-Male')
            side1Faces1 = a.instances['Epoxy-1'].faces.findAt(*Ffaces2)
            a.Surface(side1Faces=side1Faces1, name='Epoxy-Female')
            side1Faces1 = a.instances['Male-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Male-Face')
            side1Faces1 = a.instances['Female-1'].faces.findAt(*Ffaces2)
            a.Surface(side1Faces=side1Faces1, name='Female-Face')
        else:
            side1Faces1 = a.instances['Male-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Male-Face')
            side1Faces1 = a.instances['Female-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Female-Face')
    else:
        Ffaces = [((0,d_j-d_s/2,b_j/2),)]
        for i in range(numk):
            Ffaces.append(((-w_k/2,d_j-d_s-s_k*i-(d_k-d_f)/4,b_j/2), ))
            Ffaces.append(((-w_k,d_j-d_s-s_k*i-d_k/2,b_j/2), ))
            Ffaces.append(((-w_k/2,d_j-d_s-s_k*i-(3*d_k+d_f)/4,b_j/2), ))
            if i == (numk - 1):
                Ffaces.append(((0,(d_j-d_s-s_k*i-d_k)/2,b_j/2), ))
            else:
                Ffaces.append(((0,d_j-d_s-s_k*i-(d_k+s_k)/2,b_j/2), ))
        if ept != 0:
            Ffaces2 = [((0-ept,d_j-d_s/2,b_j/2),)]
            for i in range(numk):
                Ffaces2.append(((-w_k/2-ept,d_j-d_s-s_k*i-(d_k-d_f)/4,b_j/2), ))
                Ffaces2.append(((-w_k-ept,d_j-d_s-s_k*i-d_k/2,b_j/2), ))
                Ffaces2.append(((-w_k/2-ept,d_j-d_s-s_k*i-(3*d_k+d_f)/4,b_j/2), ))
                if i == (numk - 1):
                    Ffaces2.append(((0-ept,(d_j-d_s-s_k*i-d_k)/2,b_j/2), ))
            side1Faces1 = a.instances['Male-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Male-Face')
            side1Faces1 = a.instances['Female-1'].faces.findAt(*Ffaces2)
            a.Surface(side1Faces=side1Faces1, name='Female-Face')
            Ffaces.append(((-ept/2,d_j-d_s-s_k*i-ept/w_k*((d_k-d_f)/2)/2,b_j/2), ))
            Ffaces.append(((-ept/2,d_j-d_s-s_k*i-d_k+ept/w_k*((d_k-d_f)/2)/2,b_j/2), ))
            side1Faces1 = a.instances['Epoxy-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Epoxy-Male')
            Ffaces2.append(((-w_k-ept/2,d_j-d_s-s_k*i-(d_k-d_f)/2+ept/w_k*((d_k-d_f)/2)/2,b_j/2), ))
            Ffaces2.append(((-w_k-ept/2,d_j-d_s-s_k*i-(d_k+d_f)/2-ept/w_k*((d_k-d_f)/2)/2,b_j/2), ))
            side1Faces1 = a.instances['Epoxy-1'].faces.findAt(*Ffaces2)
            a.Surface(side1Faces=side1Faces1, name='Epoxy-Female')
        else: 
            side1Faces1 = a.instances['Male-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Male-Face')
            side1Faces1 = a.instances['Female-1'].faces.findAt(*Ffaces)
            a.Surface(side1Faces=side1Faces1, name='Female-Face')

    ## Add tie constraint between steel plate and PCJ
    a = mdb.models[model_name].rootAssembly
    region1=a.surfaces['CSLP-R']
    a = mdb.models[model_name].rootAssembly
    region2=a.surfaces['Female-L']
    mdb.models[model_name].Tie(name='Constraint-1', master=region1, slave=region2, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
        constraintEnforcement=SURFACE_TO_SURFACE, thickness=ON)
        
    a = mdb.models[model_name].rootAssembly
    region1=a.surfaces['CSRP-L']
    a = mdb.models[model_name].rootAssembly
    region2=a.surfaces['Male-R']
    mdb.models[model_name].Tie(name='Constraint-2', master=region1, slave=region2, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
        constraintEnforcement=SURFACE_TO_SURFACE, thickness=ON)

    a = mdb.models[model_name].rootAssembly
    region1=a.surfaces['LoadP-B']
    a = mdb.models[model_name].rootAssembly
    region2=a.surfaces['Male-T']
    mdb.models[model_name].Tie(name='Constraint-3', master=region1, slave=region2, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
        constraintEnforcement=SURFACE_TO_SURFACE, thickness=ON)
        
    ## Add embedment between rebar and PCJ    
    a = mdb.models[model_name].rootAssembly
    region1=a.sets['Set-MaleRebar']
    a = mdb.models[model_name].rootAssembly
    region2=a.sets['Set-Male']
    mdb.models[model_name].EmbeddedRegion(name='Embed-Male', embeddedRegion=region1, 
        hostRegion=region2, weightFactorTolerance=1e-06, absoluteTolerance=0.0, 
        fractionalTolerance=0.05, toleranceMethod=BOTH)
    a = mdb.models[model_name].rootAssembly
    region1=a.sets['Set-FemaleRebar']
    a = mdb.models[model_name].rootAssembly
    region2=a.sets['Set-Female']
    mdb.models[model_name].EmbeddedRegion(name='Embed-Female', 
        embeddedRegion=region1, hostRegion=region2, weightFactorTolerance=1e-06, 
        absoluteTolerance=0.0, fractionalTolerance=0.05, toleranceMethod=BOTH)
        
    ## Add coupling
    a = mdb.models[model_name].rootAssembly
    region1=a.sets['Set-RP']
    a = mdb.models[model_name].rootAssembly
    region2=a.surfaces['LoadP-T']
    mdb.models[model_name].Coupling(name='LoadCouple', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    ## Create Interface property
    mdb.models[model_name].ContactProperty('Interface')
    mdb.models[model_name].interactionProperties['Interface'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
        fri, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models[model_name].interactionProperties['Interface'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, 
        constraintEnforcementMethod=DEFAULT)
    #: The interaction property "Interface" has been created.

    a = mdb.models[model_name].rootAssembly
    region1=a.surfaces['Female-Face']
    region2=a.surfaces['Male-Face']
    if ept != 0:
        regionE1=a.surfaces['Epoxy-Female']
        regionE2=a.surfaces['Epoxy-Male']
        mdb.models[model_name].Tie(name='TE1', master=region1, slave=regionE1, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
        constraintEnforcement=SURFACE_TO_SURFACE, thickness=ON)
        mdb.models[model_name].Tie(name='TE2', master=region2, slave=regionE2, 
        positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, 
        constraintEnforcement=SURFACE_TO_SURFACE, thickness=ON)
    else:
        mdb.models[model_name].SurfaceToSurfaceContactStd(name='Interface', 
            createStepName='Initial', master=region1, slave=region2, sliding=FINITE, 
            thickness=ON, interactionProperty='Interface', adjustMethod=TOLERANCE, 
            initialClearance=OMIT, datumAxis=None, clearanceRegion=None, tied=OFF, 
            adjustTolerance=0.001)
    #: The interaction "Interface" has been created.

    #Step 5:  Create steps
    mdb.models[model_name].StaticStep(name='Contact', previous='Initial', 
        initialInc=0.02, maxInc=0.5, nlgeom=ON)
    mdb.models[model_name].StaticStep(name='Remove', previous='Contact', 
        maxNumInc=100, initialInc=0.02, maxInc=0.5)
    mdb.models[model_name].StaticStep(name='Lateral', previous='Remove', 
        maxNumInc=100, initialInc=0.01, maxInc=0.5)
    mdb.models[model_name].StaticStep(name='Vertical', previous='Lateral', 
        timePeriod=2.0, maxNumInc=10000, initialInc=0.005, minInc=2e-05, 
        maxInc=0.1)
        
    # Step 6: Load and Constraint
    ## Bottom constraint
    a = mdb.models[model_name].rootAssembly
    region = a.sets['Set-Constraint']
    mdb.models[model_name].PinnedBC(name='Bottom Fixed', createStepName='Initial', 
        region=region, localCsys=None)

    ## Smooth Amplitude
    mdb.models[model_name].SmoothStepAmplitude(name='SS-Lateral', timeSpan=STEP, 
        data=((0.0, 0.0), (1.0, 1.0)))
    mdb.models[model_name].SmoothStepAmplitude(name='SS-Vertical', timeSpan=STEP, 
        data=((0.0, 0.0), (2.0, 1.0)))
        
    ## Contact load
    a = mdb.models[model_name].rootAssembly
    region = a.surfaces['CSLP-L']
    mdb.models[model_name].Pressure(name='Contact-Left', createStepName='Contact', 
        region=region, distributionType=UNIFORM, field='', magnitude=0.02, 
        amplitude=UNSET)
        
    a = mdb.models[model_name].rootAssembly
    region = a.surfaces['CSRP-R']
    mdb.models[model_name].Pressure(name='Contact-Right', createStepName='Contact', 
        region=region, distributionType=UNIFORM, field='', magnitude=0.02, 
        amplitude=UNSET)

    ## Remove Contact load
    mdb.models[model_name].loads['Contact-Left'].deactivate('Remove')
    mdb.models[model_name].loads['Contact-Right'].deactivate('Remove')

    ## Lateral load
    a = mdb.models[model_name].rootAssembly
    region = a.surfaces['CSLP-L']
    mdb.models[model_name].Pressure(name='CSL', createStepName='Lateral', 
        region=region, distributionType=UNIFORM, field='', magnitude=cs, 
        amplitude=UNSET)
    a = mdb.models[model_name].rootAssembly
    region = a.surfaces['CSRP-R']
    mdb.models[model_name].Pressure(name='CSR', createStepName='Lateral', 
        region=region, distributionType=UNIFORM, field='', magnitude=cs, 
        amplitude=UNSET)

    ## Displacement controlled loading
    # a = mdb.models[model_name].rootAssembly
    region = a.sets['Set-RP']
    mdb.models[model_name].DisplacementBC(name='Vertical', 
        createStepName='Vertical', region=region, u1=UNSET, u2=-2.0, u3=UNSET, 
        ur1=UNSET, ur2=UNSET, ur3=UNSET, amplitude='SS-Vertical', fixed=OFF, 
        distributionType=UNIFORM, fieldName='', localCsys=None)

    # Step 7: Define monitoring points
    regionDef=mdb.models[model_name].rootAssembly.sets['Set-RP']
    mdb.models[model_name].HistoryOutputRequest(name='RP', createStepName='Lateral', 
        variables=('U2', 'RF2'), region=regionDef, sectionPoints=DEFAULT, 
        rebar=EXCLUDE)
    MElements = []
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['Male-1'].elements
    elements1 = e1.getByBoundingBox(0.0, 0.0, 0.0, size1+1.0 ,d_j+1.0 ,b_j+1.0)
    a.Set(elements=elements1, name='Set-MaleElement')
    #: The set 'Set-MaleElement' has been created (325 elements).

    regionDef=mdb.models[model_name].rootAssembly.sets['Set-MaleElement']
    mdb.models[model_name].HistoryOutputRequest(name='SPElement', 
        createStepName='Lateral', variables=('S11', 'S22', 'S12'), 
        region=regionDef, sectionPoints=DEFAULT, rebar=EXCLUDE)

    # Step 8: Create job and submit
    mdb.Job(name=model_name, model=model_name, description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', multiprocessingMode=DEFAULT, numCpus=1, numDomains=1, 
        numGPUs=1)

    mdb.jobs[model_name].writeInput(consistencyChecking=OFF)
    # #: The job input file has been written to "First.inp".

# Step 0: Creat model
mat = 'E1'
n_FDK = 9
n_FEK = 37
n_SDK = 85
n_SEK = 67
n_MDK = 31
n_MEK = 37
n_NPS4 = 4
size0 = 20.0
size1 = 10.0
size2 = 5.0
size3 = 40.0
l_eq = 20.0

database = 'SDK'
n_select = n_SDK
data = np.genfromtxt('C:\\Interest\\1. Papers\\13. Abaqus modeling of keyed dry PCJs\\1. Model calculation\\{0}.csv'.format(database), delimiter=',', skip_header=1)
for i in range(0,n_select):
    DK_model(data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6], data[i,7], data[i,8], data[i,9], data[i,10], data[i,11], data[i,12], data[i,13], data[i,14], data[i,15], data[i,16], mat, size0, size1, size2, size3, l_eq)
