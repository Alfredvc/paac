class EnvironmentCreator(object):

    def __init__(self, args):
        """
        Creates an object from which new environments can be created
        :param args:
        """

        from atari_emulator import AtariEmulator
        from ale_python_interface import ALEInterface
        filename = args.rom_path + "/" + args.game + ".bin"
        ale_int = ALEInterface()
        ale_int.loadROM(str.encode(filename))
        self.num_actions = len(ale_int.getMinimalActionSet())
        self.create_environment = lambda i: AtariEmulator(i, args)



