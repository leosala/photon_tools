import numpy as np
from time import time
import sys
import h5py
import pydoc
import image_analyses as ian


def get_dataset_tags(main_dataset):
    try:
        if "image" in main_dataset.keys():
            dataset = main_dataset["image"]
        else:
            dataset = main_dataset["data"]
        tags_list = 1e6 * main_dataset["time"]["seconds"].astype(long) + main_dataset["time"]["fiducials"]
    except:
        dataset = main_dataset
        tags_list = 1e6 * main_dataset.parent["time"]["seconds"].astype(long) + main_dataset.parent["time"]["fiducials"]
        
    return dataset, tags_list

class Analysis(object):
    """Simple container for the analysis functions to be loaded into AnalysisProcessor. At the moment, it is only used internally inside AnalysisProcessor
    """
    def __init__(self, analysis_function, arguments={}, post_analysis_function=None, name=None):
        """
        Parameters
        ----------
        analysis_function: callable function
            the main analysis function to be run on images
        arguments: dict
            arguments to analysis_function
        post_analysis_function: callable function
            function to be called only once after the analysis loop
        """

        self.function = analysis_function
        self.post_analysis_function = post_analysis_function
        self.arguments = arguments
        if name is not None:
            self.name = name
        else:
            self.name = self.function.__name__
        self.temp_arguments = {}
        self.results = {}


def images_iterator(images, chunk_size=1, mask=None, n_events=-1):
    """Standard Iterator
    ADD BLAH
    """
    i = 0
    if n_events == -1:
        n_events = images.shape[0] 

    if chunk_size >= n_events:
        chunk_size = n_events

    for i in range(0, n_events, chunk_size):
        print "Processing event %d / %d" % (i, n_events)
        end_i = min(i + chunk_size, n_events)
        if mask is not None:
            idx = np.arange(n_events)[mask]
            dset = images[i:end_i][mask[i:end_i]]
        else:
            dset = images[i:end_i]
            if dset.shape[0] == 0:
                yield None
                continue

        for j in range(dset.shape[0]):
            yield dset[j]


def images_iterator_cspad140(images, chunk_size=1, mask=None, n_events=-1):
    """Iterator over CSPAD140 images, as taken at LCLS.    
    ADD BLAH
    """

    # where to put this geometry configurations?
    pixel_width = 110 # in microns
    vertical_gap_mm = 2.3 # in mm

    i = 0
    if n_events == -1:
        n_events = images.shape[0] 

    if chunk_size >= n_events:
        chunk_size = n_events

    for i in range(0, n_events, chunk_size):
        print "Processing event %d / %d" % (i, n_events)
        end_i = min(i + chunk_size, n_events)
        if mask is not None:
            idx = np.arange(n_events)[mask]
            dset = images[i:end_i][mask[i:end_i]]
        else:
            dset = images[i:end_i]
            if dset.shape[0] == 0:
                yield None
                continue
        vertical_gap_px_arr = np.zeros((dset.shape[0], dset.shape[2], 
                                        int(round(vertical_gap_mm * 1.e+3 / pixel_width))))
        dset_glued = np.concatenate((dset.T[0].T.swapaxes(1, 2), 
                                     vertical_gap_px_arr, dset.T[1].T.swapaxes(1, 2)), axis=2)            
        for j in range(dset.shape[0]): 
            yield dset_glued[j]


class ImagesProcessor(object):
    """Simple class to perform analysis on SACLA datafiles. Due to the peculiar file 
    format in use at SACLA (each image is a single dataset), any kind of
    analysis must be performed as a loop over all images: due to this, I/O is
    not optimized, and many useful NumPy methods cannot be easily used.
    
    With this class, each image is read in memory only once, and then passed 
    to the registered methods to be analyzed. All registered functions must:
    + take as arguments at least results (dict), temp (dict), image (2D array)
    + return results, temp
    
    `results` is used to store the results produced by the function, while
    `temp` stores temporary values that must be preserved during the image loop.
    
    A simple example is:

    def get_spectra(results, temp, image, axis=0, ):
        result = image.sum(axis=axis)
        if temp["current_entry"] == 0:
            results["spectra"] = np.empty((results['n_entries'], ) + result.shape, dtype=result.dtype) 
            
        results["spectra"][temp['current_entry']] = result
        temp["current_entry"] += 1
        return results, temp
    
    In order to apply this function to all images, you need to:
    
    # create an AnalysisOnImages object
    an = ImagesProcessor()

    # load a dataset from a SACLA data file
    fname = "/home/sala/Work/Data/Sacla/ZnO/257325.h5"
    dataset_name = "detector_2d_1"
    an.set_sacla_dataset(hf, dataset_name)

    # register the function:    
    an.add_analysis(get_spectra, args={'axis': 1})

    # run the loop
    results = an.analyze_images(fname, n=1000)

    """

    def __init__(self):
        self.results = []
        self.temp = {}
        self.functions = {}
        self.datasets = []
        self.f_for_all_images = {}
        self.analyses = []
        self.available_analyses = {}
        self.available_analyses["get_histo_counts"] = (ian.get_histo_counts, None)
        self.available_analyses["get_mean_std"] = (ian.get_mean_std, ian.get_mean_std_results)
        self.available_analyses["get_projection"] = (ian.get_projection, None)
        self.available_preprocess = {}
        self.available_preprocess["set_roi"] = ian.set_roi
        self.available_preprocess["set_thr"] = ian.set_thr
        self.available_preprocess["subtract_correction"] = ian.subtract_correction
        self.available_preprocess["correct_bad_pixels"] = ian.correct_bad_pixels
        
        self.n = -1
        self.flatten_results = False
        self.preprocess_list = []
        self.dataset_name = None
        self.images_iterator = images_iterator 

    def __call__(self, dataset_file, n=-1, tags=None):
        return self.analyze_images(dataset_file, n=n, tags=tags)

    def set_images_iterator(self, func_name=None):
        """
        """
        if func_name is None:
            print "[WARNING] no images_iterator provided, doing nothing"
            return
        
        #this can be done nicer, I think
        gl_func = globals()
        if gl_func.has_key(func_name):
            self.images_iterator = gl_func[func_name]
        else:
            print sys.exc_info()
            raise RuntimeError("Images iterator function %s does not exist!" % func_name)


    def add_preprocess(self, f, label="", args={}):
        """
        Register a function to be applied to all images, before analysis (e.g. dark subtraction)
        """
        if label != "":
            f_name = label
        elif isinstance(f, str):
            f_name = f
        else:
            f_name = f.__name__

        if isinstance(f, str):
            if not self.available_preprocess.has_key(f):
                raise RuntimeError("Preprocess function %s not available, please check your code" % f)
            self.f_for_all_images[f_name] = {'f': self.available_preprocess[f], "args": args}
        else:
            self.f_for_all_images[f_name] = {'f': f, "args": args}
        print "[INFO] Preprocess %s added" % f_name
        self.preprocess_list.append(f_name)
        
    def list_preprocess(self):
        """List all loaded preprocess functions
        
        Returns
        ----------
        list :
            list of all loaded preprocess functions
        """
        return self.preprocess_list
    
    def remove_preprocess(self, label=None):
        """Remove loaded preprocess functions. If called without arguments, it removes all functions.
        
        Parameters
        ----------
        label : string
            label of the preprocess function to be removed
        """
        if label is None:
            self.f_for_all_images = {}
            self.preprocess_list = []
        else:
            del self.f_for_all_images[label]
            self.preprocess_list.remove[label] 
    
    def add_analysis(self, f, result_f=None, args={}, label=""):
        """Register a function to be run on images
        
        Parameters
        ----------
        f: callable function
            analysis function to be loaded. Must take at least results (dict), temp (dict) and image (numpy array) as input, and return just results and temp. See main class help.
        result_f: callable function
            function to be applied to the results, at the end of the loop on images.
        args: dict
            arguments for the analysis function
        label : string
            label to be assigned to analysis function
            
        Returns
        -------
            None        
        """
        
        if isinstance(f, str):
            if not self.available_analyses.has_key(f):
                raise RuntimeError("Analysis %s not available, please check your code" % f)
            analysis = Analysis(self.available_analyses[f][0], arguments=args, 
                                post_analysis_function=self.available_analyses[f][1], name=f)
        else:
            if label != "":
                analysis = Analysis(f, arguments=args, post_analysis_function=result_f, name=label)
            else:
                analysis = Analysis(f, arguments=args, post_analysis_function=result_f)
        if analysis.name in self.list_analysis():
            print "[INFO] substituting analysis %s" % analysis.name
            self.remove_analysis(label=analysis.name)
        self.analyses.append(analysis)
        #return analysis.results

    def list_analysis(self):
        """List all loaded analysis functions
        
        Returns
        ----------
        list :
            list of all loaded analysis functions
        """
        return [x.name for x in self.analyses]

    def remove_analysis(self, label=None):
        """Remove a labelled analysis. If no label is provided, all the analyses are removed
        
        Parameters
        ----------
        label : string
            Label of the analysis to be removed        
        """
        
        if label is None:
            self.analyses = []
        else:
            for an in self.analyses:
                if an.name == label:
                    self.analyses.remove(an)

    #???
    def set_dataset(self, dataset_name, remove_preprocess=True):
        """Set the name for the SACLA dataset to be analyzed        
        Parameters
        ----------
        dataset_name : string
            Name of the dataset to be added, without the trailing `/run_XXXXX/`
        remove_preprocess : bool
            Remove all the preprocess functions when setting a dataset
        """
        self.dataset_name = dataset_name
        self.results = {}
        self.temp = {}
        if remove_preprocess:
            print "[INFO] Setting a new dataset, removing stored preprocess functions. To overcome this, use remove_preprocess=False"
            self.remove_preprocess()
        
    def analyze_images(self, fname, n=-1, tags=None, chunk_size=1000):
        """Executes a loop, where the registered functions are applied to all the images
        
        Parameters
        ----------
        fname : string
            Name of HDF5 Sacla file to analyze
        n : int
            Number of events to be analyzed. If -1, then all events will be analyzed.
        tags : int list
            List of tags to be analyzed.
            
        Returns
        -------
        results: dict
            dictionary containing the results.
        """
        results = {}

        if tags == []:
            print "WARNING: emtpy tags list, returning nothing..."
            return results

        hf = h5py.File(fname, "r")

        #self.run = hf.keys()[-1]  # find a better way
        if self.dataset_name is None:
            hf.close()
            raise RuntimeError("Please provide a dataset name using the `set_sacla_dataset` method!")

        main_dataset = hf[self.dataset_name]
        dataset, tags_list = get_dataset_tags(main_dataset)

        if n != -1:
            tags_list = tags_list[:n]
        
        tags_mask = None
        dataset_indexes = np.arange(dataset.shape[0])
        if tags is not None:
            tags_mask = np.in1d(tags_list, tags, assume_unique=True)
            tags_list = tags_list[tags_mask]
            
        n_images = len(tags_list)
                        
        for analysis in self.analyses:
            analysis.results = {}
            analysis.results["n_entries"] = n_images
            analysis.results["filename"] = fname
            analysis.temp_arguments = {}
            analysis.temp_arguments["current_entry"] = 0
            analysis.temp_arguments["image_shape"] = None
            analysis.temp_arguments["image_dtype"] = None

        # loop on tags
        images_iter = self.images_iterator(dataset, chunk_size, tags_mask, n_events=n)
        
        for image_i, image in enumerate(images_iter):
            if image_i >= n_images:
                break
            
            if self.f_for_all_images != {}:
                for k in self.preprocess_list:
                    image = self.f_for_all_images[k]['f'](image, **self.f_for_all_images[k]['args'])
            
            for analysis in self.analyses:
                if image is not None and analysis.temp_arguments["image_shape"] is None:
                    analysis.temp_arguments["image_shape"] = image.shape
                    analysis.temp_arguments["image_dtype"] = image.dtype
                analysis.results, analysis.temporary_arguments = analysis.function(analysis.results, analysis.temp_arguments, image, **analysis.arguments)
            
        for analysis in self.analyses:
            if analysis.post_analysis_function is not None:
                analysis.results = analysis.post_analysis_function(analysis.results, analysis.temp_arguments)
            if self.flatten_results:
                results.update(analysis.results)
            else:
                results[analysis.name] = analysis.results

        # return also the analyzed tags
        results["tags"] = tags_list
        self.results = results
        hf.close()
        return self.results

    def print_help(self, label=""):
        """Print help for a specific analysis or preprocess function
        
        Parameters
        ----------
        label : string
            Label of the analysis or preprocess function whose help should be printed        
        """
        if label is "":
            print """\n
            ######################## 
            # Preprocess functions #
            ########################\n"""
            for f in self.available_preprocess.values():
                print pydoc.plain(pydoc.render_doc(f))

            print """\n
            ######################## 
            # Analysis  functions  #
            ########################\n"""
            for f in self.available_analyses.values():
                print pydoc.plain(pydoc.render_doc(f[0]))
        else:
            if self.available_preprocess.has_key(label):
                print """\n
                ######################## 
                # Preprocess functions #
                ########################\n"""
                print pydoc.plain(pydoc.render_doc(self.available_preprocess[label]))
            elif self.available_analyses.has_key(label):
                print """\n
                ######################## 
                # Analysis  functions  #
                ########################\n"""
                print pydoc.plain(pydoc.render_doc(self.available_analyses[label][0]))
            else:
                print "Function %s does not exist" % label
                
        
