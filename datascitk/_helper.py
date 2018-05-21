# Copyright (C) 2018 Georg Unterholzner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ==============================================================================
""" Helper functions """


def type_as_string(obj):
    """
    Takes an object and returns its type as string.

    :param obj:     Any object
    :return:        Type of the object as a string value.

    """

    return str(type(obj))[8:-2]
