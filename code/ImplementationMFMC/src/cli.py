"""CLI locale ; les commandes models/doctor fonctionnent sans dépendances ML."""
import argparse
from dataclasses import replace
import json
import sys
from .utils.config import Settings
from .utils.model_manager import ModelManager, SPECS, diagnostics
from .core.errors import FaceAnonError


def parser():
    p = argparse.ArgumentParser(prog="FaceAnon", description="Anonymisation photographique locale avec SimSwap")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Versions, checkpoints et test CUDA réel si PyTorch installé")
    models = commands.add_parser("models", help="Gestion explicite des modèles")
    models.add_argument("action", choices=["status", "download", "import"])
    models.add_argument("key", nargs="?", choices=list(SPECS))
    models.add_argument("--file")
    models.add_argument("--acknowledge-pickle-risk", action="store_true")
    bank = commands.add_parser("bank", help="Banque attestée synthétique")
    bank.add_argument("action", choices=["list", "import", "generate", "reindex"])
    bank.add_argument("--folder")
    bank.add_argument("--confirm-synthetic", action="store_true")
    bank.add_argument("--start-seed", type=int, default=0)
    bank.add_argument("--count", type=int, default=20)
    bank.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    anon = commands.add_parser("anonymize", help="Remplacement ciblé, validation obligatoire avant export")
    anon.add_argument("--reference", required=True)
    anon.add_argument("--image", required=True)
    anon.add_argument("--output", required=True)
    anon.add_argument("--reference-index", type=int)
    anon.add_argument("--select", nargs="*", type=int, help="Indices explicites des visages à remplacer (base 0)")
    anon.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    anon.add_argument("--size", type=int, choices=[224, 512], default=224)
    anon.add_argument("--match-threshold", type=float, default=.45)
    anon.add_argument("--validation-max", type=float, default=.30)
    anon.add_argument("--candidate-max", type=float, default=.30)
    anon.add_argument("--attempts", type=int, default=3)
    anon.add_argument("--mode", choices=["random", "session", "manual"], default="random")
    anon.add_argument("--identity")
    anon.add_argument("--no-segmentation", action="store_true")
    anon.add_argument("--match-luminance", action="store_true")
    anon.add_argument("--independent-model", action="store_true", help="Valider aussi avec models/evaluation/recognizer.onnx")
    anon.add_argument("--independent-validation-max", type=float, default=.30,
                      help="Seuil du second reconnaisseur, à calibrer séparément")
    anon.add_argument("--fallback", choices=["none", "mask"], default="none")
    return p


def progress(percent, message):
    print(f"[{percent:3d}%] {message}", file=sys.stderr, flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    settings = Settings()
    manager = ModelManager(settings)
    try:
        if args.command == "doctor":
            print(json.dumps({"environment": diagnostics(), "models": manager.status()}, indent=2, ensure_ascii=False))
            return 0
        if args.command == "models":
            if args.action == "status":
                print(json.dumps(manager.status(), indent=2, ensure_ascii=False))
                return 0
            if not args.key:
                raise ValueError("Spécifier le modèle")
            if args.action == "download":
                manager.download(args.key, progress, trusted_pickle=args.acknowledge_pickle_risk)
            else:
                if not args.file:
                    raise ValueError("Spécifier --file")
                manager.import_file(args.key, args.file, trusted_pickle=args.acknowledge_pickle_risk)
            return 0
        from .core.service import ProcessingService
        if args.command == "bank":
            if args.action == "list":
                path = settings.bank / "bank.json"
                data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"identities": []}
                print(json.dumps([{k: e.get(k) for k in ("key", "file", "provenance", "seed")} for e in data["identities"]],
                                 indent=2, ensure_ascii=False))
                return 0
            settings = replace(settings, device=args.device)
            pipeline = ProcessingService().pipeline(settings)
            if args.action == "import":
                if not args.folder:
                    raise ValueError("Spécifier --folder")
                count, rejected = pipeline.bank.import_folder(args.folder, synthetic_confirmed=args.confirm_synthetic, progress=progress)
            elif args.action == "generate":
                if not 1 <= args.count <= 10000 or not 0 <= args.start_seed <= 2**32-args.count:
                    raise ValueError("Nombre/seed invalide")
                from .backends.stylegan_backend import StyleGANBackend
                count, rejected = pipeline.bank.generate(StyleGANBackend(settings), range(args.start_seed, args.start_seed+args.count), progress)
            else:
                count = len(pipeline.bank.reindex(progress))
                rejected = []
            print(json.dumps({"indexed": count, "rejected": rejected}, ensure_ascii=False, indent=2))
            return 0 if count else 3
        settings = replace(settings, device=args.device, crop_size=args.size, match_threshold=args.match_threshold,
                           validation_max_similarity=args.validation_max, candidate_max_similarity=args.candidate_max,
                           independent_validation_max_similarity=args.independent_validation_max,
                           max_attempts=args.attempts, segmentation=not args.no_segmentation,
                           identity_mode=args.mode, manual_identity=args.identity, attribute_matching=args.match_luminance)
        from .utils.image_utils import load_image, save_image
        pipeline = ProcessingService().pipeline(settings)
        if args.independent_model:
            from .backends.insightface_backend import InsightFaceBackend
            independent = InsightFaceBackend(settings, ModelManager(settings).require("independent"))
            if independent.model_id == pipeline.recognizer.model_id:
                raise ValueError("Le second modèle doit être distinct du modèle de ciblage")
            pipeline.independent_recognizer = independent
        image, reference = load_image(args.image), load_image(args.reference)
        analysis = pipeline.analyze(reference, image, settings, args.reference_index, progress)
        selected = analysis.selected if args.select is None else args.select
        result = pipeline.run(analysis, settings, selected, progress)
        fallback = False
        if not result.success:
            if args.fallback != "mask" or not selected:
                print(json.dumps({"success": False, "warnings": result.warnings}, ensure_ascii=False, indent=2))
                return 3
            from .core.blending import emergency_mask
            protected = [f for i, f in enumerate(analysis.faces) if i not in selected]
            result.image = emergency_mask(image, [analysis.faces[i] for i in selected], protected)
            fallback = True
        save_image(args.output, result.image, protected_paths=[args.image, args.reference])
        from dataclasses import asdict
        print(json.dumps({"success": result.success, "fallback_mask": fallback, "detected": result.detected,
                          "recognized": result.recognized, "replaced": result.replaced,
                          "attempts": result.attempts, "identity": result.identity,
                          "faces": [asdict(r) for r in result.reports], "warnings": result.warnings,
                          "output": args.output}, ensure_ascii=False, indent=2))
        return 0 if result.success else 4
    except (FaceAnonError, ValueError, OSError, ImportError) as exc:
        print(f"FaceAnon : {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print("Installer requirements.txt avec CPython 3.10 x64 ; voir README.md.", file=sys.stderr)
        return 2
