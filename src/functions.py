# IMPORTS
import pandas as pd


# CONSTANTS

CELL_PRE = 'cell_'
PIXEL_PRE = 'pixel_'


def get_matrices(mark_area, marks_cell_associations, marks_cell_overlap):
    
    
    # prototype pixel x cell matrix for absolute area overlap of all possible pixel-cell combinations
    overlap_matrix = pd.DataFrame(index=[CELL_PRE + n for n in marks_cell_overlap.keys()], columns=[PIXEL_PRE + n for n in mark_area.keys()])

    # analogous matrix for overlap relative to each pixel area (corresponds to ablated region specific sampling proportion)
    sampling_prop_matrix = overlap_matrix.copy()

    # analogous matrix for constitution of every cell
    sampling_spec_matrix = overlap_matrix.copy()


    for cell_i in marks_cell_associations.keys():
        pixels = marks_cell_associations[cell_i]
        # get the index of the pixels as their order is the same in marks_cell_overlap
        for pixel_i, pixel_loc in enumerate(pixels):
    
            overlap_area = marks_cell_overlap[cell_i][pixel_i]
            # write absolute area overlap of current cell-pixel association to respective location in matrix
            overlap_matrix.loc[CELL_PRE + cell_i, PIXEL_PRE + pixel_loc] = overlap_area
    
    total_pixel_size = dict(zip(overlap_matrix.columns, mark_area.values()))
    sampling_prop_matrix = overlap_matrix.divide(total_pixel_size, axis=1)
    
    total_cell_coverage = overlap_matrix.sum(axis=1).replace(to_replace=0, value=1)
    sampling_spec_matrix = overlap_matrix.divide(total_cell_coverage, axis=0)

    return(overlap_matrix, sampling_prop_matrix, sampling_spec_matrix)


