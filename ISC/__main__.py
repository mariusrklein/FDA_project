"""
Main command line application of ion suppression correction tool

Author: Marius Klein (mklein@duck.com), January 2023
"""
import argparse
import os
from ISC.src.ion_suppression import IonSuppressionCorrection
from ISC.src import const

def main():
    # initialize command line application
    parser = argparse.ArgumentParser(prog="ISC",
        description = "this tool performs ion suppression correction on a single-cell" +
        " metabolomics dataset generated by SpaceM.")

    # Adding optional arguments
    parser.add_argument("path", help = "SpaceM folder path")
    parser.add_argument("-c", "--config", help = "Location and name of config file")
    parser.add_argument("-j", "--jobs", help = "set number of jobs for multiprocessing", type=int)
    parser.add_argument("-v", "--verbose", help = "produce more output", action='store_true')
    parser.add_argument("-m", "--make-config", help = "only create a default correction config " +
        "file in folder specified by path", action='store_true')

    args = parser.parse_args()

    if args.make_config:
        if args.verbose:
            print("Saving standard config file 'correction_config.json' to specified location "+
            f"'{args.path}'.")

        isc = IonSuppressionCorrection(source_path = args.path,
                config = None,
                n_jobs = 1,
                verbose = False)
        
        isc.save_config(file = os.path.join(args.path, "correction_config.json"))
        
        if args.verbose:
            print("Done. Exiting...")

        return
        
    if args.verbose:
        print(f"Running ion suppression correction with folder {get_abs_path(args.path)}")
        if args.config is not None:
            print(f"Using config file at {get_abs_path(args.config)}")
        else:
            print('Using default config:')
            print(const.CONFIG)
    if args.jobs is not None:
        print(f"Running with {args.jobs} jobs")


    # initializing instance of ion suppression correction objet
    isc = IonSuppressionCorrection(source_path = get_abs_path(args.path),
                config = get_abs_path(args.config),
                n_jobs = args.jobs,
                verbose = args.verbose)

    # perform checks on config and load metadata
    isc.prepare()
    # running correction, deconvolution, evaluation
    isc.run()

    if args.verbose:
        print("DONE! Exiting application.")
            

def get_abs_path(path: str) -> str:
        """ returns absolute path based on relative path and current directory

        Args:
            path (str): relative or absolute path

        Returns:
            str: absolute path
        """
        if os.path.isabs(path):
            return path
        
        return os.path.join(os.getcwd(), path)

if __name__ == "__main__":
    main()
    