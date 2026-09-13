from gradio_utils import get_demo


## ------------------------------------------------------ ##
debug = False

demo = get_demo()
demo.launch(server_name = "0.0.0.0", server_port = 9999, debug = debug)
