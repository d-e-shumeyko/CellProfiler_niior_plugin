#################################
#
# Imports from useful Python libraries
#
#################################

import numpy
import scipy.ndimage
import skimage
import itertools

#################################
#
# Imports from CellProfiler
#
##################################

# Reference
# 
# If you're using a technique or method that's used in this module 
# and has a publication attached to it, please include the paper link below.
# Otherwise, remove the line below and remove the "References" section from __doc__.
#

cite_paper_link = "https://doi.org/10.1016/1047-3203(90)90014-M"

__doc__ = """\
niior MLM Database
=============

This plugin is intended to take in primary and tertiary objects identifed by 
CellProfiler and output the objects as 28x28 pixel images in order to create 
a database of neurons and nuclei for future use in MLM development.

**ImageTemplate** is an example image processing module. It's recommended to
put a brief description of this module here and go into more detail below.

This is an example of a module that takes one image as an input and
produces a second image for downstream processing. You can use this as
a starting point for your own module: rename this file and put it in your
plugins directory.

The text you see here will be displayed as the help for your module, formatted
as `reStructuredText <http://docutils.sourceforge.net/rst.html>`_.

Note whether or not this module supports 3D image data and respects masks.
A module which respects masks applies an image's mask and operates only on
the image data not obscured by the mask. Update the table below to indicate 
which image processing features this module supports.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

SaveCroppedObjects


What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

Are there any assumptions about input data someone using this module
should be made aware of? For example, is there a strict requirement that
image data be single-channel, or that the foreground is brighter than
the background? Describe any assumptions here.

This section can be omitted if there is no requirement on the input.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

Describe the output of this module. This is necessary if the output is
more complex than a single image. For example, if there is data displayed
over the image then describe what the data represents.

This section can be omitted if there is no specialized output.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Describe the measurements made by this module, if any. Typically, measurements
are described in the following format:

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

**Measurement category:**

-  *MeasurementName*: A brief description of the measurement.
-  *MeasurementName*: A brief description of the measurement.

This section can be omitted if no measurements are made by this module.

Technical notes
^^^^^^^^^^^^^^^

Include implementation details or notes here. Additionally provide any 
other background information about this module, including definitions
or adopted conventions. Information which may be too specific to fit into
the general description should be provided here.

Omit this section if there is no technical information to mention.

References
^^^^^^^^^^

Provide citations here, if appropriate. Citations are formatted as a list and,
wherever possible, include a link to the original work. For example,

-  Meyer F, Beucher S (1990) “Morphological segmentation.” *J Visual
   Communication and Image Representation* 1, 21-46.
   {cite_paper_link}
"""

#
# Constants
#
# It's good programming practice to replace things like strings with
# constants if they will appear more than once in your program. That way,
# if someone wants to change the text, that text will change everywhere.
# Also, you can't misspell it by accident.
#
import os.path
import logging

from cellprofiler_core.image import Image
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.object import Objects
from cellprofiler_core.module import Module, ImageProcessing
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME, get_default_colormap
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting import Divider, HiddenCount, SettingsGroup, Binary
from cellprofiler_core.setting.subscriber import LabelSubscriber, ImageSubscriber, FileImageSubscriber
from cellprofiler_core.constants.measurement import C_FILE_NAME
from cellprofiler_core.setting.text import Float, Integer, ImageName, Text, Directory, LabelName
from cellprofiler_library.modules import savecroppedobjects

LOGGER = logging.getLogger(__name__)

O_PNG = "png"
SAVE_PER_OBJECT = "Images"
R_TO_SIZE = "Resize by specifying desired final dimensions"
C_MANUAL = "Manual"
I_NEAREST_NEIGHBOR = "Nearest Neighbor"
"""Parent (seed) relationship of input objects to output objects"""
R_PARENT = "Parent"

#
# The module class.
#
# Your module should "inherit" from cellprofiler_core.module.Module, or a
# subclass of cellprofiler_core.module.Module. This module inherits from
# cellprofiler_core.module.ImageProcessing, which is the base class for
# image processing modules. Image processing modules take an image as
# input and output an image.
#
# This module will use the methods from cellprofiler_core.module.ImageProcessing
# unless you re-implement them. You can let cellprofiler_core.module.ImageProcessing
# do most of the work and implement only what you need.
#
# Other classes you can inherit from are:
#
# -  cellprofiler_core.module.image_segmentation.ImageSegmentation: modules which take an image
#    as input and output a segmentation (objects) should inherit from this
#    class.
# -  cellprofiler_core.module.image_segmentation.ObjectProcessing: modules which operate on objects
#    should inherit from this class. These are modules that take objects as
#    input and output new objects.
#
class niiorMLMDatabasePlugin(ObjectProcessing):
    #

    
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    # This module's category is "Image Processing" which is defined
    # by its superclass.
    #
    module_name = "niiorMLMDatabasePlugin"
    category = "Object Processing"
    variable_revision_number = 1

    #
    # Citation
    #
    # If you're using a technique or method that's used in this module 
    # and has a publication attached to it, please include the paper link below.
    # Edit accordingly and add the link for the paper as "https://doi.org/XXXX".
    # If no citation is necessary, remove the "doi" dictionary below. 
    #

    doi = {"Please cite the following when using ImageTemplate:": 'https://doi.org/10.1016/1047-3203(90)90014-M', 
           "If you're also using specific technique X, cite:": 'https://doi.org/10.1016/1047-3203(90)90014-M'}
    #
    # "create_settings" is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler_core.settings for
    # settings you can use.
    #
   
    
    
    def create_settings(self):
        #
        # The superclass (ImageProcessing) defines two
        # settings for image input and output:
        #
        # -  x_name: an ImageNameSubscriber which "subscribes" to all
        #    ImageNameProviders in prior modules. Modules before yours will
        #    put images into CellProfiler. The ImageNameSubscriber gives
        #    your user a list of these images which can then be used as inputs
        #    in your module.
        # -  y_name: an ImageName makes the image available to subsequent
        #    modules.
        super(niiorMLMDatabasePlugin, self).create_settings()
#         self.x_name.text = "Select the input objects"
#         self.x_name.doc = """\
# ToDo
# """
        #
        # reST help that gets displayed when the user presses the
        # help button to the right of the edit box.
        #
        # The superclass defines some generic help test. You can add
        # module-specific help text by modifying the setting's "doc"
        # string.
        #
        
        #
        # Here's a choice box - the user gets a drop-down list of what
        # can be done.
        #
        
        #Settings I added
        
    
        self.file_format = Choice(
            "Saved file format",
            [O_PNG],
            value=O_PNG,
            doc="""\
            ToDo""".format(
                O_PNG=O_PNG
            ),
        )

        
        self.objects_name = LabelSubscriber(
            "Objects", 
            doc="Select the objects to export as per-object crops.",
        )
        self.image_name = ImageSubscriber(
            "Image to crop", 
            doc="Select the image to crop",
        )
        self.directory = Directory(
            "Directory",
            doc="Enter the directory where object crops are saved.",
            value=DEFAULT_OUTPUT_FOLDER_NAME,
        )
        self.file_image_name = FileImageSubscriber(
            "Select image name to use as a prefix",
            "None",
            doc="""\
Select an image loaded using **NamesAndTypes**. The original filename
will be used as the prefix for the output filename."""
        )
       
        self.specific_width = Integer(
            "Width (x) of the final image",
            28,
            minval=1,
            doc="""\

Enter the desired width of the final image, in pixels.""",
        )

        self.specific_height = Integer(
            "Height (y) of the final image",
            28,
            minval=1,
            doc="""\

Enter the desired height of the final image, in pixels.""",
        )
        
        # Resize portion
        
        
        self.size_method = Choice(
            "Resizing method",
            R_TO_SIZE,
            doc="""\
The following options are available:
-  *Resize by specifying desired final dimensions:* Enter the new height and width of the resized image, in units of pixels.""",
        )
        
        self.use_manual = Choice(
            "Method to specify the dimensions",
            C_MANUAL,
            doc="""\
*(Used only if resizing by specifying the dimensions)*

You have two options on how to resize your image:

-  *{C_MANUAL}:* Specify the height and width of the output image.

            """.format(
                **{"C_MANUAL":  C_MANUAL}
            ),
        )

        self.specific_width = Integer(
            "Width (x) of the final image",
            28,
            minval=1,
            doc="""\
*(Used only if resizing by specifying desired final dimensions)*

Enter the desired width of the final image, in pixels.""",
        )

        self.specific_height = Integer(
            "Height (y) of the final image",
            28,
            minval=1,
            doc="""\
*(Used only if resizing by specifying desired final dimensions)*

Enter the desired height of the final image, in pixels.""",
        )

        self.specific_planes = Integer(
            "# of planes (z) in the final image",
            1,
            minval=1,
            doc="""\
*(Used only if resizing by specifying desired final dimensions)*

Enter the desired number of planes in the final image.""",
        )
        
        self.interpolation = Choice(
            "",
            I_NEAREST_NEIGHBOR
            
        )
        
        
        #End of my settings
        

    def settings(self):
        #
        # The superclass's "settings" method returns [self.x_name, self.y_name],
        # which are the input and output image settings.
        
        settings = super(niiorMLMDatabasePlugin, self).settings()
       
        # Append additional settings here.
        return (
            settings
            + [         
                self.image_name,
                self.file_image_name,
                self.objects_name,
                self.directory,
                self.specific_height,
                self.specific_width,
                self.size_method,
                self.use_manual,
                self.specific_width,
                self.specific_height,
                self.specific_planes,
                self.interpolation

            ]
            
        )

    #
    # "visible_settings" tells CellProfiler which settings should be
    # displayed and in what order.
    #
    # You don't have to implement "visible_settings" - if you delete
    # visible_settings, CellProfiler will use "settings" to pick settings
    # for display.
    #
    def visible_settings(self):
        #
        # The superclass's "visible_settings" method returns [self.x_name,
        # self.y_name], which are the input and output image settings.
        #
        visible_settings = super(niiorMLMDatabasePlugin, self).visible_settings()
        visible_settings += [
                 self.image_name,
                 self.objects_name,
                 self.directory,
                 self.file_image_name
                ]
        return visible_settings 

        # Configure the visibility of additional settings below.
        
    def add_image(self, can_remove=True):
        group = SettingsGroup()

        if can_remove:
            group.append("divider", Divider(line=False))

        group.append(
            "input_image_name",
            ImageSubscriber(
                "Select the additional image?",
                "None",
                doc="""\
What is the name of the additional image to resize? This image will be
resized with the same settings as the first image.""",
            ),
        )

        group.append(
            "output_image_name",
            ImageName(
                "Name the output image",
                "ResizedBlue",
                doc="What is the name of the additional resized image?",
            ),
        )

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove above image", self.additional_images, group
                ),
            )

        self.additional_images.append(group)
       

        
    #
    # CellProfiler calls "run" on each image set in your pipeline.
    #
    
    def display(self, workspace, figure):
        figure.set_subplots((1, 1))

        #figure.subplot_table(0, 0, [["\n".join(workspace.display_data.filenames)]])


    
    def run(self, workspace):
        #
        # The superclass's "run" method handles retreiving the input image
        # and saving the output image. Module-specific behavior is defined
        # by setting "self.function", defined in this module. "self.function"
        # is called after retrieving the input image and before saving
        # the output image.
        #
        # The first argument of "self.function" is always the input image
        # data (as a numpy array). The remaining arguments are the values of
        # the module settings as they are returned from "settings" (excluding
        # "self.y_data", or the output image).
        #
       
        
        # super(niiorMLMDatabasePlugin, self).run(workspace)
        objects = workspace.object_set.get_objects(self.objects_name.value)
        input_objects = objects.segmented
        output_objects = ""
        directory = self.directory.get_absolute_path(workspace.measurements)
        input_objects_name = self.objects_name.value
        
        input_filename = workspace.measurements.get_current_measurement("Image", self.source_file_name_feature)
        input_filename = os.path.splitext(input_filename)[0]
        
        images = workspace.image_set
        x = images.get_image(self.image_name.value).pixel_data        
        
        exp_options = {
            "png": "png"
        }

        
        
        
        filenames = saveresizedcroppedobjects(
            input_objects,
            save_dir=directory,
            export_as=SAVE_PER_OBJECT,
            input_image=x,
            file_format=exp_options[self.file_format.value],
            save_names = {"input_filename": input_filename, "input_objects_name": input_objects_name}            
        )
        apply_resize(self, workspace, input_objects_name, output_objects)
    
        if self.show_window:
            workspace.display_data.filenames = filenames

    #
    # "volumetric" indicates whether or not this module supports 3D images.
    # The "gradient_image" function is inherently 2D, and we've noted this
    # in the documentation for the module. Explicitly return False here
    # to indicate that 3D images are not supported.
    #
    def volumetric(self):
        return False

    @property
    def source_file_name_feature(self):
        """The file name measurement for the exemplar disk image"""
        return "_".join((C_FILE_NAME, self.file_image_name.value))
#
# This is the function that gets called during "run" to create the output image.
# The first parameter must be the input image data. The remaining parameters are
# the additional settings defined in "settings", in the order they are returned.
#
# This function must return the output image data (as a numpy array).
#
# def gradient_image(pixels, gradient_choice, automatic_smoothing, scale):
#     #
#     # Get the smoothing parameter
#     #
#     if automatic_smoothing:
#         # Pick the mode of the power spectrum - obviously this
#         # is pretty hokey, not intended to really find a good number.
#         #
#         fft = numpy.fft.fft2(pixels)
#         power2 = numpy.sqrt((fft * fft.conjugate()).real)
#         mode = numpy.argwhere(power2 == power2.max())[0]
#         scale = numpy.sqrt(numpy.sum((mode + 0.5) ** 2))

#     gradient_magnitude = scipy.ndimage.gaussian_gradient_magnitude(pixels, scale)

#     if gradient_choice == GRADIENT_MAGNITUDE:
#         gradient_image = gradient_magnitude
#     else:
#         # Image data is indexed by rows and columns, with a given point located at
#         # position (row, column). Here, x represents the column coordinate (at index 1)
#         # and y represents the row coordinate (at index 0).
#         #
#         # You can learn more about image coordinate systems here:
#         # http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
#         x = scipy.ndimage.correlate1d(gradient_magnitude, [-1, 0, 1], 1)
#         y = scipy.ndimage.correlate1d(gradient_magnitude, [-1, 0, 1], 0)
#         norm = numpy.sqrt(x ** 2 + y ** 2)
#         if gradient_choice == GRADIENT_DIRECTION_X:
#             gradient_image = 0.5 + x / norm / 2
#         else:
#             gradient_image = 0.5 + y / norm / 2

#     return gradient_image


def resize_and_save_object_image_crops(
    input_image,
    input_objects,
    save_dir,
    file_format="tiff8",
    nested_save=False,
    save_names = {"input_filename": None, "input_objects_name": None},
    volumetric=False
    ):
    """
    For a given input_objects array, save crops for each 
    object of the provided input_image.
    """
    # Build save paths
    if nested_save:
        if not save_names["input_filename"] and not save_names["input_objects_name"]:
            raise ValueError("Must provide a save_names['input_filename'] or save_names['input_objects_name'] for nested save.")
        save_path = os.path.join(
            save_dir, 
            save_names["input_filename"] if save_names["input_filename"] else save_names["input_objects_name"],
            )
    else:
        save_path = save_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    unique_labels = numpy.unique(input_objects)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    labels = input_objects

    if len(input_image.shape) == len(input_objects.shape) + 1 and not volumetric:
        labels = numpy.repeat(
            labels[:, :, numpy.newaxis], input_image.shape[-1], axis=2
        )

    # Construct filename
    save_filename = f"{save_names['input_filename']+'_' if save_names['input_filename'] else ''}{save_names['input_objects_name']+'_' if save_names['input_objects_name'] else ''}"

    save_filenames = []
    
    for label in unique_labels:
        file_extension = "tiff" if "tiff" in file_format else "png"

        label_save_filename = os.path.join(save_path, save_filename + f"{label}.{file_extension}")
        save_filenames.append(label_save_filename)
        mask_in = labels == label
        properties = skimage.measure.regionprops(
                mask_in.astype(int), intensity_image=input_image
            )
        mask = properties[0].intensity_image
        
        if file_format.casefold() == "png":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                check_contrast=False
            )
        elif file_format.casefold() == "tiff8":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                compression=(8,6),
                check_contrast=False,
            )
        elif file_format.casefold() == "tiff16":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_uint(mask),
                compression=(8,6),
                check_contrast=False,
            )
        else:
            raise ValueError(f"{file_format} not in 'png', 'tiff8', or 'tiff16'")
    
    return save_filenames

def saveresizedcroppedobjects(
    input_objects,
    save_dir,
    export_as="masks",
    input_image=None,
    file_format="tiff8",
    nested_save=False,
    save_names={"input_filename": None, "input_objects_name": None},
    volumetric=False
    ):
    if export_as.casefold() in ("image", "images"):
        filenames = resize_and_save_object_image_crops(
            input_image=input_image,
            input_objects=input_objects,
            save_dir=save_dir,
            file_format=file_format,
            nested_save=nested_save,
            save_names=save_names,
            volumetric=volumetric
        )
    
    return 

# Resize methods

def resized_shape(self, image, workspace):
    
        image = object_to_image(self, workspace)

        image_pixels = image.pixel_data

        shape = numpy.array(image_pixels.shape).astype(float)


       

        
        
        height = self.specific_height.value
        width = self.specific_width.value
        if image.volumetric:
            planes = self.specific_planes.value
        

        new_shape = []

        if image.volumetric:
            new_shape += [planes]

        new_shape += [height, width]

        if image.multichannel:
            new_shape += [shape[-1]]

        return numpy.asarray(new_shape)
def spline_order(self):
        if self.interpolation.value == I_NEAREST_NEIGHBOR:
            return 0
        else:
            LOGGER.warning(
                'interpolation error'
            )
def apply_resize(self, workspace, input_image_name, output_image_name):
        image =  object_to_image(self, workspace)
        directory = self.directory.get_absolute_path(workspace.measurements)
        image_pixels = image.pixel_data

        new_shape = resized_shape(self, image, workspace)

        order = spline_order(self)

        
        output_pixels = skimage.transform.resize(
                image_pixels, new_shape, order=order, mode="symmetric"
            )

        if image.multichannel and len(new_shape) > image.dimensions:
            new_shape = new_shape[:-1]

        mask = skimage.transform.resize(image.mask, new_shape, order=0, mode="constant")

        mask = skimage.img_as_bool(mask)

        if image.has_crop_mask:
            cropping = skimage.transform.resize(
                image.crop_mask, new_shape, order=0, mode="constant"
            )

            cropping = skimage.img_as_bool(cropping)
        else:
            cropping = None

        output_image = Image(
            output_pixels,
            parent_image=image,
            mask=mask,
            crop_mask=cropping,
            dimensions=image.dimensions,
        )

        workspace.image_set.add(output_image_name, output_image)
        save_resized_images(image, output_image, directory, file_format="png", save_names={"input_filename": None, "input_objects_name": None})
        
        



        if self.show_window:
            if hasattr(workspace.display_data, "input_images"):
                workspace.display_data.multichannel += [image.multichannel]
                workspace.display_data.input_images += [image.pixel_data]
                workspace.display_data.output_images += [output_image.pixel_data]
                workspace.display_data.input_image_names += [input_image_name]
                workspace.display_data.output_image_names += [output_image_name]
            else:
                workspace.display_data.dimensions = image.dimensions
                workspace.display_data.multichannel = [image.multichannel]
                workspace.display_data.input_images = [image.pixel_data]
                workspace.display_data.output_images = [output_image.pixel_data]
                workspace.display_data.input_image_names = [input_image_name]
                workspace.display_data.output_image_names = [output_image_name]
                
                
                
                
# to save resized crops
def save_resized_images(input_image, 
                        output_image, save_dir, 
                        file_format="png", 
                        save_names = {"input_filename": None, "input_objects_name": None},):
    os.makedirs(save_dir, exist_ok=True)
    
    unique_labels = numpy.unique(output_image)
    
    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    labels = unique_labels
    # if len(output_image.pixel_data) == len(input_image.pixel_data) + 1 :
       # flat = list(itertools.chain.from_iterable(labels))
    labels = numpy.repeat(
        labels[ :, numpy.newaxis], input_image.pixel_data[-1], axis=2
        )



 
    
    save_filename = f"{save_names['input_filename']+'_' if save_names['input_filename'] else ''}{save_names['input_objects_name']+'_' if save_names['input_objects_name'] else ''}"

    save_filenames = []
    for label in unique_labels:
        file_extension = "tiff" if "tiff" in file_format else "png"

        label_save_filename = os.path.join(save_dir, save_filename + f"{label}.{file_extension}")
        save_filenames.append(label_save_filename)
        mask_in = labels == label
        properties = skimage.measure.regionprops(
                mask_in.astype(int), # intensity_image=input_image
            )
        mask = properties[0].intensity_image
        
        if file_format.casefold() == "png":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                check_contrast=False
            )
        else:
            raise ValueError(f"{file_format} not in 'png', 'tiff8', or 'tiff16'")
    return save_filenames
              
                
                
                
# object to image

def object_to_image (self, workspace):
    
    objects = workspace.object_set.get_objects(self.objects_name.value)
    # objects = workspace.object_set.get_objects
    alpha = numpy.zeros(objects.shape)
    convert = True
    
    
    
    image_name = ImageName
    
    pixel_data = numpy.zeros(objects.shape + (3,))
    
    cm_name = get_default_colormap()
    
    mask = alpha > 0
    
    pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, numpy.newaxis]
    
    image = Image(
            pixel_data,
            parent_image=objects.parent_image,
            convert=convert,
            dimensions=len(objects.shape),
        )
    workspace.image_set.add(self.image_name.value, image)
    return image