class Coordinator:
    """run in controller side, the runners run in dockers with GPUs"""

    def __init__(self):
        pass

    def add_runner(self):
        """add_runner will create kernel folders and change the code"""

    def push(self):
        "Push the code to server/kagger docker"

    def push_listen(self):
        self.push()
        self._get_result()

    def _get_result(self):
        """use the message queue, just use this right after push, listen for
        result, debug local first"""
        "use RE change source code to add the log collector"
