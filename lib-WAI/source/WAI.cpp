#include <WAI.h>

void WAI::WAI::setMode(ModeType modeType)
{
    if (_mode) delete _mode;

    switch (modeType)
    {
        case ModeType_ORB_SLAM2:
        {
            if (_sensors.find(SensorType_Camera) == _sensors.end())
            {
                printf("Cannot switch to mode ORB_SLAM2 since camera sensor is not activated. Please call activate sensor with AstrolabeSensorType_Camera first.\n");
                return;
            }

            _mode = new ModeOrbSlam2((SensorCamera*)_sensors[SensorType_Camera],
                                     false,
                                     false,
                                     false,
                                     false);
        }
        break;

        case ModeType_Aruco:
        {
            if (_sensors.find(SensorType_Camera) == _sensors.end())
            {
                printf("Cannot switch to mode Aruco since camera sensor is not activated. Please call activate sensor with AstrolabeSensorType_Camera first.\n");
                return;
            }

            _mode = new ModeAruco((SensorCamera*)_sensors[SensorType_Camera]);
        }
        break;

        case ModeType_None:
        default:
        {
            // TODO(jan): error handling
        }
        break;
    }
}

void WAI::WAI::activateSensor(SensorType sensorType, void* sensorInfo)
{
    if (_sensors.find(sensorType) != _sensors.end())
    {
        printf("sensor with type %i already activated.\n",
               sensorType);
        return;
    }

    switch (sensorType)
    {
        case SensorType_Camera:
        {
            _sensors[SensorType_Camera] = new SensorCamera((CameraCalibration*)sensorInfo);
        }
        break;

        case SensorType_None:
        default:
        {
            // TODO(jan): error handling
        }
        break;
    }
}

void WAI::WAI::updateSensor(SensorType sensorType, void* value)
{
    if (_sensors.find(sensorType) == _sensors.end())
    {
        printf("Trying to update a non-existent sensor %i. Call activate sensor with the appropriate sensortype.\n",
               sensorType);
        return;
    }

    _sensors[sensorType]->update(value);
}

bool WAI::WAI::whereAmI(M4x4* pose)
{
    wai_assert(_mode && "No mode set. Call setMode before calling whereAmI.");

    bool result = _mode->getPose(pose);

    return result;
}
