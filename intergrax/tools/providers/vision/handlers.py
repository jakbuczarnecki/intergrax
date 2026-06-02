# © Artur Czarnecki. All rights reserved.

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.vision.contracts import VisionDetectInput, VisionDetectOutput
from intergrax.tools.providers.vision.service import vision_detect


class VisionDetectHandler(ServiceToolHandler[VisionDetectInput, VisionDetectOutput]):
    _service = vision_detect
