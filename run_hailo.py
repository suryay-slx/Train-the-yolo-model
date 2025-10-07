import argparse
import os
import sys
from pathlib import Path
import numpy as np
import queue
import threading
from PIL import Image
from typing import List
from typing import List, Generator, Optional, Tuple, Dict
from pathlib import Path
from functools import partial
import queue
import numpy as np
from PIL import Image
from hailo_platform import (HEF, VDevice,
                            FormatType, HailoSchedulingAlgorithm)
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')

# Add the parent directory to the system path to access utils module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class HailoAsyncInference:
    def __init__(
        self, hef_path: str, input_queue: queue.Queue,
        output_queue: queue.Queue, batch_size: int = 1,
        input_type: Optional[str] = None, output_type: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize the HailoAsyncInference class with the provided HEF model 
        file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file.
            input_queue (queue.Queue): Queue from which to pull input frames 
                                       for inference.
            output_queue (queue.Queue): Queue to hold the inference results.
            batch_size (int): Batch size for inference. Defaults to 1.
            input_type (Optional[str]): Format type of the input stream. 
                                        Possible values: 'UINT8', 'UINT16'.
            output_type (Optional[str]): Format type of the output stream. 
                                         Possible values: 'UINT8', 'UINT16', 'FLOAT32'.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        params = VDevice.create_params()     
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)      
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.output_type = output_type

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Optional[str] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type of all of them.

        Args:
            output_type (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(getattr(FormatType, output_type)) 
    def callback(
        self, completion_info, bindings_list: list, processed_batch: list
    ) -> None:
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the 
                             inference task.
            bindings_list (list): List of binding objects containing input 
                                  and output buffers.
            processed_batch (list): The processed batch of images.
        """
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(bindings.output(name).get_buffer(), axis=0)
                    for name in bindings._output_names
                }               
            self.output_queue.put((processed_batch[i], result))

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )
    def get_hef(self) -> HEF:
        """
        Get the object's HEF file
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        """
        Run asynchronous inference on the Hailo device, processing batches 
        from the input queue.

        Batches are fetched from the input queue until a sentinel value 
        (None) is encountered.
        """
        with self.infer_model.configure() as configured_infer_model:
            while True:
                batch_frames = self.input_queue.get()  
                # Get the tuple (processed_batch, batch_array) from the queue
                if batch_frames is None:
                    break  # Sentinel value to stop the inference loop

                bindings_list = []
                for frame in batch_frames:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000)
                job = configured_infer_model.run_async(
                    bindings_list, partial(
                        self.callback, processed_batch=batch_frames,
                        bindings_list=bindings_list
                    )
                )
            job.wait(10000)  # Wait for the last job
    
    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
            for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape, 
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
            for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )
         

def divide_list_to_batches(
    images_list: List[Image.Image], batch_size: int
) -> Generator[List[Image.Image], None, None]:
    """
    Divide the list of images into batches.

    Args:
        images_list (List[Image.Image]): List of images.
        batch_size (int): Number of images in each batch.

    Returns:
        Generator[List[Image.Image], None, None]: Generator yielding batches 
                                                  of images.
    """
    for i in range(0, len(images_list), batch_size):
        yield images_list[i: i + batch_size]
        


def enqueue_images(
    images: List[Image.Image], 
    batch_size: int, 
    input_queue: queue.Queue, 
    width: int, 
    height: int, 
) -> None:
    """
    Preprocess and enqueue images into the input queue as they are ready.

    Args:
        images (List[Image.Image]): List of PIL.Image.Image objects.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
    """
    for batch in divide_list_to_batches(images, batch_size):
        input_queue.put(batch)

    input_queue.put(None)  # Add sentinel value to signal end of input


def process_output(
    output_queue: queue.Queue, 
    width: int, 
    height: int, 
) -> None:
    """
    Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        output_path (Path): Path to save the output images.
        width (int): Image width.
        height (int): Image height.
    """
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received
        
        processed_image, infer_results = result
        print(infer_results)
        image_id += 1
    
    output_queue.task_done()  # Indicate that processing is complete


def infer(
    images: List[Image.Image], 
    net_path: str, 
    batch_size: int, 
) -> None:
    """
    Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (List[Image.Image]): List of images to process.
        net_path (str): Path to the HEF model file.
        labels_path (str): Path to a text file containing labels.
        batch_size (int): Number of images per batch.
    """
    
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, batch_size
    )
    height, width, _ = hailo_inference.get_input_shape()

    enqueue_thread = threading.Thread(
        target=enqueue_images, 
        args=(images, batch_size, input_queue, width, height)
    )
    process_thread = threading.Thread(
        target=process_output, 
        args=(output_queue, width, height)
    )
    
    enqueue_thread.start()
    process_thread.start()
    
    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    process_thread.join()




def main() -> None:
    model_name = "resnet18"
    batchSize = 8
    images =  np.random.randn(batchSize, 3, 224, 224).astype(np.float32)

    # Start the inference
    infer(images, batchSize)


if __name__ == "__main__":
    main()
