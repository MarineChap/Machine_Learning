#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 21:38:41 2018

@author: marinechap
"""


import matplotlib.colors as cl

def display_graph(indep_var, nb_cluster, model, title_str):
    
    data_set = dict(
        mode = "markers",
        name = "dataset",
        type = "scatter3d",  
        marker = dict(
            size = 4,
            color = 'black',
            ),
         x = indep_var[:, 0], y = indep_var[:, 1], z = indep_var[:, 2]
    )
    
    data_tot = [data_set]
    
    colorNames = list(cl._colors_full_map.values())
    
    for cluster_index in range(0, nb_cluster):
        cluster_data = dict(
            alphahull = 3,
            name = 'cluster {0}'.format(cluster_index),
            opacity = 0.1,
            type = "mesh3d",
            color = colorNames[cluster_index],    
            x = indep_var[model == cluster_index, 0] , 
            y = indep_var[model == cluster_index, 1] ,
            z = indep_var[model == cluster_index, 2] 
        )
        
        data_tot.append(cluster_data)
    
    
    layout = dict(
        title = title_str,
        legend = dict(
                      x=0,
                      y=1,
                      traceorder='normal',
                      font=dict(
                                family='sans-serif',
                                size=12,
                                color='#000'
                                ),
            bgcolor='#E2E2E2',
            bordercolor='#FFFFFF',
            borderwidth=2
        ),
        showlegend= True,
        scene = dict(
            xaxis = dict( 
                    zeroline=False, 
                    title = 'age'),
            yaxis = dict( 
                    zeroline=False, 
                    title = 'Annual income (K$)',),
            zaxis = dict( 
                    zeroline=False,
                    title = 'Spending score')
            )
    )


    return data_tot, layout