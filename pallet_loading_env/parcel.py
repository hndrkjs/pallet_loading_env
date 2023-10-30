class Parcel():
    """
    This class is used to represent a parcel in the environment.
    It stores information about the parcel, such as its size, and its weight.
    """

    def __init__(self, length, width, height, weight, compression_index):
        """
        Initializes the parcel.

        Args:
            length (int): The length of the parcel.
            width (int): The width of the parcel.
            height (int): The height of the parcel.
            weight (int): The weight of the parcel.
            compression_index (int): The compression index of the parcel.
        """
        self.__length = length
        self.__width = width
        self.__height = height
        self.__weight = weight
        self.__compression_index = compression_index

    @property
    def volume(self):
        """
        Returns the volume of the parcel.
        """
        return self.__length * self.__width * self.__height
    
    @property
    def base_area(self):
        """
        Returns the base area of the parcel.
        """
        return self.__length * self.__width

    
    @property
    def size(self):
        """
        Returns the size of the parcel.
        """
        return (self.__length, self.__width)
    
    @property
    def length(self):
        """
        Returns the length of the parcel.
        """
        return self.__length
    
    @property
    def width(self):
        """
        Returns the width of the parcel.
        """
        return self.__width
    
    @property
    def height(self):
        """
        Returns the height of the parcel.
        """
        return self.__height
    
    @property
    def weight(self):
        """
        Returns the weight of the parcel.
        """
        return self.__weight
    
    @property
    def compression_index(self):
        """
        Returns the compression index of the parcel.
        """
        return self.__compression_index
    
    @length.setter
    def length(self, length):
        """
        Sets the length of the parcel.
        """
        self.__length = length
    
    @width.setter
    def width(self, width):
        """
        Sets the width of the parcel.
        """
        self.__width = width